extern crate gfx_hal as hal;
extern crate gfx_memory;
extern crate imgui;
#[macro_use]
extern crate failure;
#[macro_use]
extern crate memoffset;

use std::mem;

use gfx_memory::{
    Block, Factory, FactoryError, Item, MemoryAllocator, SmartAllocator, Type,
};
use hal::memory::Properties;
use hal::pso::{PipelineStage, Rect};
use hal::{
    buffer, command, device, format, image, memory, pass, pso, queue, Backend,
    DescriptorPool, Device, Primitive, QueueGroup,
};
use imgui::{DrawData, ImDrawIdx, ImDrawVert, ImGui, Ui};

type SmartBlock<B> = <SmartAllocator<B> as MemoryAllocator<B>>::Block;

#[derive(Clone, Debug, Fail)]
pub enum Error {
    #[fail(display = "memory factory error")]
    FactoryError(#[cause] FactoryError),

    #[fail(display = "image view error")]
    ImageViewError(#[cause] image::ViewError),

    #[fail(display = "execution error")]
    ExecutionError(#[cause] hal::error::HostExecutionError),

    #[fail(display = "pipeline allocation error")]
    PipelineAllocationError(#[cause] pso::AllocationError),

    #[fail(display = "pipeline creation error")]
    PipelineCreationError(#[cause] pso::CreationError),

    #[fail(display = "mapping error")]
    MappingError(#[cause] hal::mapping::Error),

    // ShaderError doesn't implement Fail, for some reason
    #[fail(display = "shader error: {:?}", _0)]
    ShaderError(device::ShaderError),
}

macro_rules! impl_from {
    ($t:ty, $other:ty, $variant:path) => {
        impl From<$other> for $t {
            fn from(err: $other) -> $t {
                $variant(err)
            }
        }
    };
}

impl_from!(Error, FactoryError, Error::FactoryError);
impl_from!(Error, image::ViewError, Error::ImageViewError);
impl_from!(Error, hal::error::HostExecutionError, Error::ExecutionError);
impl_from!(Error, pso::AllocationError, Error::PipelineAllocationError);
impl_from!(Error, pso::CreationError, Error::PipelineCreationError);
impl_from!(Error, device::ShaderError, Error::ShaderError);
impl_from!(Error, hal::mapping::Error, Error::MappingError);

struct Buffer<B: Backend, T> {
    length: usize,
    buffer: Item<B::Buffer, SmartBlock<B>>,
    mapped: *mut u8,
    _phantom: std::marker::PhantomData<T>,
}

pub struct Renderer<B: Backend> {
    sampler: B::Sampler,
    index_buffer: Option<Buffer<B, ImDrawIdx>>,
    vertex_buffer: Option<Buffer<B, ImDrawVert>>,
    image: Item<B::Image, SmartBlock<B>>,
    image_view: B::ImageView,
    descriptor_pool: B::DescriptorPool,
    descriptor_set_layout: B::DescriptorSetLayout,
    descriptor_set: B::DescriptorSet,
    pipeline: B::GraphicsPipeline,
    pipeline_layout: B::PipelineLayout,
}

impl<B: Backend, T> Buffer<B, T> {
    /// Allocates a new buffer
    fn new(
        length: usize,
        usage: buffer::Usage,
        device: &B::Device,
        allocator: &mut SmartAllocator<B>,
    ) -> Result<Buffer<B, T>, Error> {
        let buffer = allocator.create_buffer(
            device,
            (Type::General, Properties::CPU_VISIBLE),
            (length * mem::size_of::<T>()) as u64,
            usage,
        )?;

        let mapped =
            device.map_memory(buffer.block().memory(), buffer.block().range())?;

        Ok(Buffer {
            length,
            buffer,
            mapped,
            _phantom: Default::default(),
        })
    }

    /// Copies data into the buffer
    fn update(&mut self, data: &[T], offset: usize)
    where
        T: Clone,
    {
        assert!(self.length >= data.len() + offset);
        unsafe {
            let dest =
                self.mapped.offset((offset * mem::size_of::<T>()) as isize);
            let src = &data[0];
            std::ptr::copy_nonoverlapping(src, dest as *mut T, data.len());
        }
    }

    /// Destroys the buffer
    fn destroy(self, device: &B::Device, allocator: &mut SmartAllocator<B>) {
        device.unmap_memory(self.buffer.block().memory());
        allocator.destroy_buffer(device, self.buffer);
    }

    /// Flush memory changes to syncrhonize
    fn flush(&self, device: &B::Device) {
        device.flush_mapped_memory_ranges(&[(
            self.buffer.block().memory(),
            self.buffer.block().range(),
        )]);
    }
}

impl<B: Backend> Renderer<B> {
    /// Initializes the renderer.
    pub fn new<C>(
        imgui: &mut ImGui,
        device: &B::Device,
        render_pass: &B::RenderPass,
        command_pool: &mut hal::CommandPool<B, C>,
        queue_group: &mut QueueGroup<B, C>,
        allocator: &mut SmartAllocator<B>,
    ) -> Result<Renderer<B>, Error>
    where
        // yuck
        (queue::Transfer, C): queue::capability::Upper,
        C: queue::Supports<queue::Transfer>,
        C: queue::Supports<
            <(queue::Transfer, C) as queue::capability::Upper>::Result,
        >,
    {
        // Copy texture
        let (image, image_view, image_staging) = imgui
            .prepare_texture::<_, Result<_, Error>>(|handle| {
                let size = u64::from(handle.width * handle.height * 4);

                // Create target image
                let kind = image::Kind::D2(handle.width, handle.height, 1, 1);
                let format = format::Format::Rgba8Unorm;
                let image = allocator.create_image(
                    device,
                    (Type::General, Properties::DEVICE_LOCAL),
                    kind,
                    1,
                    format,
                    image::Tiling::Optimal,
                    image::Usage::SAMPLED | image::Usage::TRANSFER_DST,
                    image::StorageFlags::empty(),
                )?;

                let subresource_range = image::SubresourceRange {
                    aspects: format::Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                };

                let image_view = device.create_image_view(
                    image.raw(),
                    image::ViewKind::D2,
                    format,
                    format::Swizzle::NO,
                    subresource_range.clone(),
                )?;

                // Create staging buffer
                let staging_buffer = allocator.create_buffer(
                    device,
                    (
                        Type::ShortLived,
                        Properties::CPU_VISIBLE | Properties::COHERENT,
                    ),
                    size,
                    buffer::Usage::TRANSFER_SRC,
                )?;

                // Coppy data into the mapped staging buffer
                {
                    let mut map = device.acquire_mapping_writer(
                        staging_buffer.block().memory(),
                        staging_buffer.block().range(),
                    )?;
                    map.clone_from_slice(handle.pixels);
                    device.release_mapping_writer(map);
                }

                // Build a command buffer to copy data
                let submit = {
                    let mut cbuf = command_pool.acquire_command_buffer(false);

                    // Copy staging buffer to the image
                    let image_barrier = memory::Barrier::Image {
                        states: (
                            image::Access::empty(),
                            image::Layout::Undefined,
                        )
                            ..(
                                image::Access::TRANSFER_WRITE,
                                image::Layout::TransferDstOptimal,
                            ),
                        target: image.raw(),
                        range: subresource_range.clone(),
                    };

                    cbuf.pipeline_barrier(
                        PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                        memory::Dependencies::empty(),
                        &[image_barrier],
                    );

                    cbuf.copy_buffer_to_image(
                        staging_buffer.raw(),
                        image.raw(),
                        image::Layout::TransferDstOptimal,
                        &[command::BufferImageCopy {
                            buffer_offset: 0,
                            buffer_width: handle.width,
                            buffer_height: handle.height,
                            image_layers: image::SubresourceLayers {
                                aspects: format::Aspects::COLOR,
                                level: 0,
                                layers: 0..1,
                            },
                            image_offset: image::Offset { x: 0, y: 0, z: 0 },
                            image_extent: image::Extent {
                                width: handle.width,
                                height: handle.height,
                                depth: 1,
                            },
                        }],
                    );

                    let image_barrier = memory::Barrier::Image {
                        states: (
                            image::Access::TRANSFER_WRITE,
                            image::Layout::TransferDstOptimal,
                        )
                            ..(
                                image::Access::SHADER_READ,
                                image::Layout::ShaderReadOnlyOptimal,
                            ),
                        target: image.raw(),
                        range: subresource_range.clone(),
                    };
                    cbuf.pipeline_barrier(
                        PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                        memory::Dependencies::empty(),
                        &[image_barrier],
                    );

                    cbuf.finish()
                };

                // Submit to the queue
                let submission = queue::Submission::new().submit(Some(submit));
                queue_group.queues[0].submit(submission, None);

                Ok((image, image_view, staging_buffer))
            })?;

        // Create font sampler
        let sampler = device.create_sampler(image::SamplerInfo::new(
            image::Filter::Linear,
            image::WrapMode::Clamp,
        ));

        // Create descriptor set
        let descriptor_set_layout = device.create_descriptor_set_layout(
            &[pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: pso::DescriptorType::CombinedImageSampler,
                count: 1,
                stage_flags: pso::ShaderStageFlags::FRAGMENT,
                immutable_samplers: true,
            }],
            Some(&sampler),
        );

        let mut descriptor_pool = device.create_descriptor_pool(
            1,
            &[pso::DescriptorRangeDesc {
                ty: pso::DescriptorType::CombinedImageSampler,
                count: 1,
            }],
        );

        let descriptor_set =
            descriptor_pool.allocate_set(&descriptor_set_layout)?;

        {
            let write = pso::DescriptorSetWrite {
                set: &descriptor_set,
                binding: 0,
                array_offset: 0,
                descriptors: &[pso::Descriptor::CombinedImageSampler(
                    &image_view,
                    image::Layout::ShaderReadOnlyOptimal,
                    &sampler,
                )],
            };
            device.write_descriptor_sets(Some(write));
        }

        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(
            Some(&descriptor_set_layout),
            // 4 is the magic number because there are two 2d vectors
            &[(pso::ShaderStageFlags::VERTEX, 0..4)],
        );

        // Create shaders
        let vs_module = {
            let spirv = include_bytes!("../shaders/ui.vert.spirv");
            device.create_shader_module(&spirv[..])?
        };
        let fs_module = {
            let spirv = include_bytes!("../shaders/ui.frag.spirv");
            device.create_shader_module(&spirv[..])?
        };

        // Create pipeline
        let pipeline = {
            let vs_entry = pso::EntryPoint {
                entry: "main",
                module: &vs_module,
                specialization: &[],
            };
            let fs_entry = pso::EntryPoint {
                entry: "main",
                module: &fs_module,
                specialization: &[],
            };

            let shader_entries = pso::GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            // Create render pass
            let subpass = pass::Subpass {
                index: 0,
                main_pass: render_pass,
            };

            let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
                shader_entries,
                Primitive::TriangleList,
                pso::Rasterizer {
                    cull_face: pso::Face::NONE,
                    ..pso::Rasterizer::FILL
                },
                &pipeline_layout,
                subpass,
            );

            // Enable blending
            pipeline_desc.blender.targets.push(pso::ColorBlendDesc(
                pso::ColorMask::ALL,
                pso::BlendState::ALPHA,
            ));

            // Set up vertex buffer
            pipeline_desc.vertex_buffers.push(pso::VertexBufferDesc {
                binding: 0,
                stride: mem::size_of::<ImDrawVert>() as u32,
                rate: 0,
            });

            // Set up vertex attributes
            // Position
            pipeline_desc.attributes.push(pso::AttributeDesc {
                location: 0,
                binding: 0,
                element: pso::Element {
                    format: format::Format::Rg32Float,
                    offset: offset_of!(ImDrawVert, pos) as u32,
                },
            });
            // UV
            pipeline_desc.attributes.push(pso::AttributeDesc {
                location: 1,
                binding: 0,
                element: pso::Element {
                    format: format::Format::Rg32Float,
                    offset: offset_of!(ImDrawVert, uv) as u32,
                },
            });
            // Color
            pipeline_desc.attributes.push(pso::AttributeDesc {
                location: 2,
                binding: 0,
                element: pso::Element {
                    format: format::Format::Rgba8Unorm,
                    offset: offset_of!(ImDrawVert, col) as u32,
                },
            });

            // Create pipeline
            device.create_graphics_pipeline(&pipeline_desc)?
        };

        // Clean up shaders
        device.destroy_shader_module(vs_module);
        device.destroy_shader_module(fs_module);

        // Wait until all transfers have finished
        queue_group.queues[0].wait_idle()?;

        // Destroy any temporary resources
        allocator.destroy_buffer(device, image_staging);

        Ok(Renderer {
            sampler,
            vertex_buffer: None,
            index_buffer: None,
            image,
            image_view,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            pipeline,
            pipeline_layout,
        })
    }

    fn draw(
        &mut self,
        ui: &Ui,
        draw_data: &DrawData,
        pass: &mut command::RenderSubpassCommon<B>,
        device: &B::Device,
        allocator: &mut SmartAllocator<B>,
    ) -> Result<(), Error> {
        // Possibly reallocate buffers
        if self
            .vertex_buffer
            .as_ref()
            .map(|buffer| buffer.length < draw_data.total_vtx_count())
            .unwrap_or(true)
        {
            let buffer = Buffer::new(
                draw_data.total_vtx_count(),
                buffer::Usage::VERTEX,
                device,
                allocator,
            )?;
            if let Some(old) =
                mem::replace(&mut self.vertex_buffer, Some(buffer))
            {
                old.destroy(device, allocator);
            }
        }
        let vertex_buffer = self.vertex_buffer.as_mut().unwrap();
        let mut vertex_offset = 0;

        if self
            .index_buffer
            .as_ref()
            .map(|buffer| buffer.length < draw_data.total_idx_count())
            .unwrap_or(true)
        {
            let buffer = Buffer::new(
                draw_data.total_idx_count(),
                buffer::Usage::INDEX,
                device,
                allocator,
            )?;
            if let Some(old) =
                mem::replace(&mut self.index_buffer, Some(buffer))
            {
                old.destroy(device, allocator);
            }
        }
        let index_buffer = self.index_buffer.as_mut().unwrap();
        let mut index_offset = 0;

        // Bind pipeline
        pass.bind_graphics_descriptor_sets(
            &self.pipeline_layout,
            0,
            Some(&self.descriptor_set),
            None as Option<u32>,
        );
        pass.bind_graphics_pipeline(&self.pipeline);

        // Bind vertex and index buffers
        pass.bind_vertex_buffers(
            0,
            pso::VertexBufferSet(vec![(vertex_buffer.buffer.raw(), 0)]),
        );
        pass.bind_index_buffer(buffer::IndexBufferView {
            buffer: index_buffer.buffer.raw(),
            offset: 0,
            index_type: hal::IndexType::U16,
        });

        let (width, height) = ui.imgui().display_size();

        // Set up viewport
        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: width as u16,
                h: height as u16,
            },
            depth: 0.0..1.0,
        };
        pass.set_viewports(0, &[viewport]);

        // Set push constants
        let push_constants = [
            // scale
            2.0 / width,
            2.0 / height,
            //offset
            -1.0,
            -1.0,
        ];
        // yikes
        let push_constants: [u32; 4] =
            unsafe { std::mem::transmute(push_constants) };
        pass.push_graphics_constants(
            &self.pipeline_layout,
            pso::ShaderStageFlags::VERTEX,
            0,
            &push_constants,
        );

        // Iterate over drawlists
        for list in draw_data {
            // Update vertex and index buffers
            vertex_buffer.update(list.vtx_buffer, vertex_offset);
            index_buffer.update(list.idx_buffer, index_offset);

            for cmd in list.cmd_buffer.iter() {
                // Calculate the scissor
                let scissor = Rect {
                    x: cmd.clip_rect.x as u16,
                    y: cmd.clip_rect.y as u16,
                    w: (cmd.clip_rect.z - cmd.clip_rect.x) as u16,
                    h: (cmd.clip_rect.w - cmd.clip_rect.y) as u16,
                };
                pass.set_scissors(0, &[scissor]);

                // Actually draw things
                pass.draw_indexed(
                    index_offset as u32..index_offset as u32 + cmd.elem_count,
                    vertex_offset as i32,
                    0..1,
                );

                index_offset += cmd.elem_count as usize;
            }

            // Increment offsets
            vertex_offset += list.vtx_buffer.len();
        }

        vertex_buffer.flush(device);
        index_buffer.flush(device);

        Ok(())
    }

    /// Renders a frame.
    pub fn render(
        &mut self,
        ui: Ui,
        render_pass: &mut command::RenderSubpassCommon<B>,
        device: &B::Device,
        allocator: &mut SmartAllocator<B>,
    ) -> Result<(), Error> {
        ui.render(|ui, draw_data| {
            self.draw(ui, &draw_data, render_pass, device, allocator)
        })?;
        Ok(())
    }

    /// Destroys all used objects.
    pub fn destroy(
        mut self,
        device: &B::Device,
        allocator: &mut SmartAllocator<B>,
    ) {
        allocator.destroy_image(device, self.image);
        device.destroy_image_view(self.image_view);
        device.destroy_sampler(self.sampler);
        self.descriptor_pool.reset();
        device.destroy_descriptor_pool(self.descriptor_pool);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout);
        device.destroy_graphics_pipeline(self.pipeline);
        device.destroy_pipeline_layout(self.pipeline_layout);
        if let Some(buffer) = self.index_buffer {
            buffer.destroy(device, allocator);
        }
        if let Some(buffer) = self.vertex_buffer {
            buffer.destroy(device, allocator);
        }
    }
}
