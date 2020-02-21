use std::{ptr, time::Instant};

use hal::prelude::*;
use hal::{command, format, image, memory, pass, pool, pso, queue, window};
use imgui::im_str;
use imgui_winit_support::{HiDpiMode, WinitPlatform};

fn main() {
    env_logger::init();

    let mut imgui = imgui::Context::create();

    let events_loop = winit::event_loop::EventLoop::new();

    let window = winit::window::Window::new(&events_loop).expect("failed to create winit window");

    let mut platform = WinitPlatform::init(&mut imgui);
    platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default);

    let instance = back::Instance::create("imgui-gfx-hal", 1).expect("failed to create instance");

    let mut surface = unsafe {
        instance
            .create_surface(&window)
            .expect("failed to create surface")
    };

    let mut adapters = instance.enumerate_adapters().into_iter();

    let (adapter, device, mut queue_group) = {
        let adapter = adapters.next().expect("No suitable adapter found");

        // Build a new device and associated command queues
        let family = adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .expect("failed to find queue family supporting surface");

        let mut gpu = unsafe {
            adapter
                .physical_device
                .open(&[(family, &[1.0])], hal::Features::empty())
                .expect("failed to open physical device")
        };

        let queue_group = gpu
            .queue_groups
            .pop()
            .expect("no queue groups available for gpu");
        let device = gpu.device;
        (adapter, device, queue_group)
    };

    let mut command_pool = unsafe {
        device
            .create_command_pool(queue_group.family, pool::CommandPoolCreateFlags::empty())
            .expect("Can't create command pool")
    };

    let caps = surface.capabilities(&adapter.physical_device);
    let formats = surface.supported_formats(&adapter.physical_device);
    let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
        formats
            .iter()
            .find(|format| format.base_format().1 == format::ChannelType::Srgb)
            .cloned()
            .unwrap_or(formats[0])
    });

    let extent = match caps.current_extent {
        Some(e) => e,
        None => {
            let window_size = window.inner_size();
            let mut extent = hal::window::Extent2D {
                width: window_size.width as u32,
                height: window_size.height as u32,
            };

            extent.width = extent
                .width
                .max(caps.extents.start().width)
                .min(caps.extents.end().width);
            extent.height = extent
                .height
                .max(caps.extents.start().height)
                .min(caps.extents.end().height);

            extent
        }
    };

    let swap_config = window::SwapchainConfig::new(
        extent.width,
        extent.height,
        format,
        *caps.image_count.start(),
    )
    .with_image_usage(image::Usage::COLOR_ATTACHMENT);
    let (mut swap_chain, backbuffer) = unsafe {
        device
            .create_swapchain(&mut surface, swap_config, None)
            .unwrap()
    };

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
            stages: pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT
                ..pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            accesses: image::Access::empty()
                ..(image::Access::COLOR_ATTACHMENT_READ | image::Access::COLOR_ATTACHMENT_WRITE),
            flags: memory::Dependencies::empty(),
        };

        unsafe {
            device
                .create_render_pass(&[attachment], &[subpass], &[dependency])
                .unwrap()
        }
    };

    let mut renderer = imgui_gfx_hal::Renderer::<back::Backend>::new(
        &mut imgui,
        &device,
        &adapter.physical_device,
        &render_pass,
        0,
        1,
        &mut command_pool,
        &mut queue_group.queues[0],
    )
    .expect("failed to setup renderer");

    let (mut frame_images, mut framebuffers): (
        std::vec::Vec<(
            <back::Backend as hal::Backend>::Image,
            <back::Backend as hal::Backend>::ImageView,
        )>,
        Vec<<back::Backend as hal::Backend>::Framebuffer>,
    ) = {
        let extent = image::Extent {
            width: extent.width as _,
            height: extent.height as _,
            depth: 1,
        };

        let pairs = backbuffer
            .into_iter()
            .map(|image| unsafe {
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
            .map(|&(_, ref rtv)| unsafe {
                device
                    .create_framebuffer(&render_pass, Some(rtv), extent)
                    .unwrap()
            })
            .collect();
        (pairs, fbos)
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
    let present_semaphore = device.create_semaphore().unwrap();
    let mut frame_fence = device.create_fence(false).unwrap();

    let mut last_frame = Instant::now();
    let mut opened = true;

    let mut cmd_buffer = unsafe { command_pool.allocate_one(command::Level::Primary) };

    events_loop.run(move |event, _, control_flow| {
        use winit::{
            event::{Event, WindowEvent},
            event_loop::ControlFlow,
        };

        *control_flow = ControlFlow::Poll;

        match event {
            Event::NewEvents(_) => {
                // other application-specific logic
                last_frame = imgui.io_mut().update_delta_time(last_frame);
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                device.wait_idle().unwrap();

                unsafe {
                    device.destroy_command_pool(ptr::read(&command_pool));
                    device.destroy_fence(ptr::read(&frame_fence));
                    device.destroy_semaphore(ptr::read(&frame_semaphore));
                    device.destroy_semaphore(ptr::read(&present_semaphore));
                    device.destroy_render_pass(ptr::read(&render_pass));

                    for framebuffer in framebuffers.drain(..) {
                        device.destroy_framebuffer(framebuffer);
                    }

                    for (_, rtv) in frame_images.drain(..) {
                        device.destroy_image_view(rtv);
                    }

                    device.destroy_swapchain(ptr::read(&swap_chain));
                    renderer.destroy(&device);
                }

                *control_flow = ControlFlow::Exit;
                return;
            }
            Event::MainEventsCleared => {
                // other application-specific logic
                platform
                    .prepare_frame(imgui.io_mut(), &window) // step 4
                    .expect("Failed to prepare frame");
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let ui = imgui.frame();
                // application-specific rendering *under the UI*

                ui.show_demo_window(&mut opened);
                let texture_id = unsafe { std::mem::transmute::<usize, imgui::TextureId>(0) };

                imgui::Window::new(im_str!("Image Test"))
                    .position([20.0, 20.0], imgui::Condition::Appearing)
                    .size([700.0, 200.0], imgui::Condition::Appearing)
                    .build(&ui, || {
                        ui.text(im_str!("Hello world!"));
                        imgui::Image::new(texture_id, [100.0f32, 100.0f32]).build(&ui);
                        ui.separator();
                        let mouse_pos = ui.io().mouse_pos;
                        ui.text(im_str!(
                            "Mouse Position: ({:.1},{:.1})",
                            mouse_pos[0],
                            mouse_pos[1]
                        ));
                    });

                platform.prepare_render(&ui, &window);

                let frame: window::SwapImageIndex = unsafe {
                    match swap_chain.acquire_image(!0, Some(&mut frame_semaphore), None) {
                        Ok(i) => i.0,
                        Err(err) => panic!("problem: {:?}", err),
                    }
                };

                unsafe {
                    cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

                    {
                        cmd_buffer.begin_render_pass(
                            &render_pass,
                            &framebuffers[frame as usize],
                            viewport.rect,
                            &[command::ClearValue {
                                color: command::ClearColor {
                                    float32: [0.2, 0.2, 0.2, 1.0],
                                },
                            }],
                            command::SubpassContents::Inline,
                        );

                        // Frame is always 0, since no double buffering.
                        renderer
                            .render(ui, 0, &mut cmd_buffer, &device, &adapter.physical_device)
                            .unwrap();
                    }

                    cmd_buffer.finish();

                    let submission = queue::Submission {
                        command_buffers: Some(&cmd_buffer),
                        wait_semaphores: Some((
                            &frame_semaphore,
                            pso::PipelineStage::BOTTOM_OF_PIPE,
                        )),
                        signal_semaphores: Some(&present_semaphore),
                    };

                    queue_group.queues[0].submit(submission, Some(&mut frame_fence));

                    if let Err(_) = swap_chain.present(
                        &mut queue_group.queues[0],
                        frame,
                        Some(&present_semaphore),
                    ) {
                        panic!("problem presenting swapchain");
                    }

                    // Wait for the command buffer to be done.
                    device.wait_for_fence(&frame_fence, !0).unwrap();
                    device.reset_fence(&frame_fence).unwrap();
                    command_pool.reset(true);
                }

                // application-specific rendering *over the UI*
            }
            event => {
                platform.handle_event(imgui.io_mut(), &window, &event);
            }
        }
    });
}
