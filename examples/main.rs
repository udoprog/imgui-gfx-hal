extern crate env_logger;
extern crate gfx_backend_vulkan as back;
extern crate gfx_hal as hal;
extern crate imgui;
extern crate imgui_gfx_hal;
extern crate winit;

use std::time::Instant;

use hal::format::ChannelType;
use hal::pso::PipelineStage;
use hal::{
    command, format, image, pass, pool, pso, Backbuffer, Device, FrameSync,
    Instance, Submission, Surface, Swapchain, SwapchainConfig,
};
use imgui::{FrameSize, ImGui};

#[derive(Copy, Clone, PartialEq, Debug, Default)]
struct MouseState {
    pos: (i32, i32),
    pressed: (bool, bool, bool),
    wheel: f32,
}

fn main() {
    env_logger::init();

    let mut imgui = ImGui::init();

    let mut events_loop = winit::EventsLoop::new();

    let window = winit::Window::new(&events_loop).unwrap();
    let instance = back::Instance::create("imgui-gfx-hal", 1);
    let mut surface = instance.create_surface(&window);
    let mut adapters = instance.enumerate_adapters().into_iter();

    let (adapter, device, mut queue_group) = loop {
        let adapter = adapters.next().expect("No suitable adapter found");
        match adapter.open_with::<_, gfx_hal::Graphics>(1, |family| {
            surface.supports_queue_family(family)
        }) {
            Ok((device, queue_group)) => break (adapter, device, queue_group),
            Err(_) => (),
        }
    };
    let physical_device = &adapter.physical_device;

    let mut command_pool = device
        .create_command_pool_typed(
            &queue_group,
            pool::CommandPoolCreateFlags::empty(),
            16,
        )
        .unwrap();

    let (caps, formats, _) = surface.compatibility(&adapter.physical_device);
    let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
        formats
            .iter()
            .find(|format| format.base_format().1 == ChannelType::Srgb)
            .cloned()
            .unwrap_or(formats[0])
    });

    let extent = match caps.current_extent {
        Some(e) => e,
        None => {
            let window_size = window.get_inner_size().unwrap();
            let mut extent = hal::window::Extent2D {
                width: window_size.width as u32,
                height: window_size.height as u32,
            };

            extent.width = extent
                .width
                .max(caps.extents.start.width)
                .min(caps.extents.end.width);
            extent.height = extent
                .height
                .max(caps.extents.start.height)
                .min(caps.extents.end.height);

            extent
        }
    };

    let swap_config = SwapchainConfig::new(
        extent.width,
        extent.height,
        format,
        caps.image_count.start,
    )
    .with_image_usage(image::Usage::COLOR_ATTACHMENT);
    let (mut swap_chain, backbuffer) = device
        .create_swapchain(&mut surface, swap_config, None)
        .unwrap();

    let render_pass = {
        let attachment = pass::Attachment {
            format: Some(format),
            samples: 1,
            ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::Clear,
                pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: pass::AttachmentOps::DONT_CARE,
            layouts: image::Layout::Undefined..image::Layout::Present,
        };

        let subpass = pass::SubpassDesc {
            colors: &[(0, image::Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let dependency = pass::SubpassDependency {
            passes: pass::SubpassRef::External..pass::SubpassRef::Pass(0),
            stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT
                ..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            accesses: image::Access::empty()
                ..(image::Access::COLOR_ATTACHMENT_READ
                    | image::Access::COLOR_ATTACHMENT_WRITE),
        };

        device
            .create_render_pass(&[attachment], &[subpass], &[dependency])
            .unwrap()
    };

    let mut renderer = imgui_gfx_hal::Renderer::new(
        &mut imgui,
        &device,
        physical_device,
        &render_pass,
        0,
        1,
        &mut command_pool,
        &mut queue_group.queues[0],
    )
    .unwrap();

    let (frame_images, framebuffers) = match backbuffer {
        Backbuffer::Images(images) => {
            let extent = image::Extent {
                width: extent.width as _,
                height: extent.height as _,
                depth: 1,
            };
            let pairs = images
                .into_iter()
                .map(|image| {
                    let rtv = device
                        .create_image_view(
                            &image,
                            image::ViewKind::D2,
                            format,
                            format::Swizzle::NO,
                            image::SubresourceRange {
                                aspects: format::Aspects::COLOR,
                                levels: 0..1,
                                layers: 0..1,
                            },
                        )
                        .unwrap();
                    (image, rtv)
                })
                .collect::<Vec<_>>();
            let fbos = pairs
                .iter()
                .map(|&(_, ref rtv)| {
                    device
                        .create_framebuffer(&render_pass, Some(rtv), extent)
                        .unwrap()
                })
                .collect();
            (pairs, fbos)
        }
        Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo]),
    };

    let viewport = pso::Viewport {
        rect: pso::Rect {
            x: 0,
            y: 0,
            w: extent.width as i16,
            h: extent.height as i16,
        },
        depth: 0.0..1.0,
    };

    let mut frame_semaphore = device.create_semaphore().unwrap();
    let mut frame_fence = device.create_fence(false).unwrap();

    let mut last_frame = Instant::now();
    let mut running = true;
    let mut opened = true;
    let mut mouse_state = MouseState::default();

    while running {
        events_loop.poll_events(|event| {
            use winit::ElementState::Pressed;
            use winit::WindowEvent::*;
            use winit::{Event, MouseButton, MouseScrollDelta, TouchPhase};

            if let Event::WindowEvent { event, .. } = event {
                match event {
                    CloseRequested => running = false,
                    KeyboardInput { input, .. } => {
                        use winit::VirtualKeyCode as Key;

                        let pressed = input.state == Pressed;
                        match input.virtual_keycode {
                            Some(Key::Tab) => imgui.set_key(0, pressed),
                            Some(Key::Left) => imgui.set_key(1, pressed),
                            Some(Key::Right) => imgui.set_key(2, pressed),
                            Some(Key::Up) => imgui.set_key(3, pressed),
                            Some(Key::Down) => imgui.set_key(4, pressed),
                            Some(Key::PageUp) => imgui.set_key(5, pressed),
                            Some(Key::PageDown) => imgui.set_key(6, pressed),
                            Some(Key::Home) => imgui.set_key(7, pressed),
                            Some(Key::End) => imgui.set_key(8, pressed),
                            Some(Key::Delete) => imgui.set_key(9, pressed),
                            Some(Key::Back) => imgui.set_key(10, pressed),
                            Some(Key::Return) => imgui.set_key(11, pressed),
                            Some(Key::Escape) => imgui.set_key(12, pressed),
                            Some(Key::A) => imgui.set_key(13, pressed),
                            Some(Key::C) => imgui.set_key(14, pressed),
                            Some(Key::V) => imgui.set_key(15, pressed),
                            Some(Key::X) => imgui.set_key(16, pressed),
                            Some(Key::Y) => imgui.set_key(17, pressed),
                            Some(Key::Z) => imgui.set_key(18, pressed),
                            Some(Key::LControl) | Some(Key::RControl) => {
                                imgui.set_key_ctrl(pressed)
                            }
                            Some(Key::LShift) | Some(Key::RShift) => {
                                imgui.set_key_shift(pressed)
                            }
                            Some(Key::LAlt) | Some(Key::RAlt) => {
                                imgui.set_key_alt(pressed)
                            }
                            Some(Key::LWin) | Some(Key::RWin) => {
                                imgui.set_key_super(pressed)
                            }
                            _ => {}
                        }
                    }
                    CursorMoved { position: pos, .. } => {
                        let (x, y) = pos.into();
                        mouse_state.pos = (x, y);
                    }
                    MouseInput { state, button, .. } => match button {
                        MouseButton::Left => {
                            mouse_state.pressed.0 = state == Pressed
                        }
                        MouseButton::Right => {
                            mouse_state.pressed.1 = state == Pressed
                        }
                        MouseButton::Middle => {
                            mouse_state.pressed.2 = state == Pressed
                        }
                        _ => {}
                    },
                    MouseWheel {
                        delta: MouseScrollDelta::LineDelta(_, y),
                        phase: TouchPhase::Moved,
                        ..
                    } => mouse_state.wheel = y,
                    MouseWheel {
                        delta: MouseScrollDelta::PixelDelta(pos),
                        phase: TouchPhase::Moved,
                        ..
                    } => mouse_state.wheel = pos.y as f32,
                    ReceivedCharacter(c) => imgui.add_input_character(c),
                    _ => (),
                }
            }
        });

        let scale = imgui.display_framebuffer_scale();
        imgui.set_mouse_pos(
            mouse_state.pos.0 as f32 / scale.0,
            mouse_state.pos.1 as f32 / scale.1,
        );
        imgui.set_mouse_down([
            mouse_state.pressed.0,
            mouse_state.pressed.1,
            mouse_state.pressed.2,
            false,
            false,
        ]);
        imgui.set_mouse_wheel(mouse_state.wheel / scale.1);
        mouse_state.wheel = 0.0;

        let now = Instant::now();
        let delta = now - last_frame;
        let delta_s = delta.as_secs() as f32
            + delta.subsec_nanos() as f32 / 1_000_000_000.0;
        last_frame = now;

        device.reset_fence(&frame_fence).unwrap();
        command_pool.reset();
        let frame: hal::SwapImageIndex = {
            match swap_chain
                .acquire_image(!0, FrameSync::Semaphore(&mut frame_semaphore))
            {
                Ok(i) => i,
                Err(err) => panic!("problem: {:?}", err),
            }
        };

        let physical_size = window
            .get_inner_size()
            .unwrap()
            .to_physical(window.get_hidpi_factor());
        let hidpi_factor = window.get_hidpi_factor().round();
        let logical_size = physical_size.to_logical(hidpi_factor);
        let frame_size = FrameSize {
            logical_size: logical_size.into(),
            hidpi_factor,
        };
        let ui = imgui.frame(frame_size, delta_s);
        ui.show_demo_window(&mut opened);

        let submit = {
            let mut cmd_buffer = command_pool.acquire_command_buffer(false);

            {
                let mut encoder = cmd_buffer.begin_render_pass_inline(
                    &render_pass,
                    &framebuffers[frame as usize],
                    viewport.rect,
                    &[command::ClearValue::Color(command::ClearColor::Float(
                        [0.2, 0.2, 0.2, 1.0],
                    ))],
                );

                // Frame is always 0, since no double buffering.
                renderer
                    .render(ui, 0, &mut encoder, &device, &physical_device)
                    .unwrap();
            }

            cmd_buffer.finish()
        };

        let submission = Submission::new()
            .wait_on(&[(&frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)])
            .submit(Some(submit));
        queue_group.queues[0].submit(submission, Some(&mut frame_fence));

        device.wait_for_fence(&frame_fence, !0).unwrap();

        if let Err(()) =
            swap_chain.present(&mut queue_group.queues[0], frame, &[])
        {
            panic!("problem presenting swapchain");
        }
    }

    device.destroy_command_pool(command_pool.into_raw());
    device.destroy_fence(frame_fence);
    device.destroy_semaphore(frame_semaphore);
    device.destroy_render_pass(render_pass);
    for framebuffer in framebuffers {
        device.destroy_framebuffer(framebuffer);
    }
    for (_, rtv) in frame_images {
        device.destroy_image_view(rtv);
    }
    device.destroy_swapchain(swap_chain);
    renderer.destroy(&device);
}
