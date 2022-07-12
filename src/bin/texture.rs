use std::{
	default::Default,
	ffi::CString,
	io::Cursor,
	mem::{self},
	num::NonZeroU32,
	ops::Deref
};

use examples::*;
use vulkayes_core::{
	ash::{util::*, vk},
	memory::device::MappingAccessResult,
	resource::{
		buffer::{params::BufferAllocatorParams, Buffer},
		image::{
			params::{ImageAllocatorParams, ImageSize, ImageViewRange, MipmapLevels},
			view::ImageView,
			Image
		}
	}
};

vulkayes_core::offsetable_struct! {
	#[derive(Copy, Clone, Debug)]
	struct Vertex {
		pos: [f32; 4],
		uv: [f32; 2]
	} repr(C) as VertexOffsets
}

#[derive(Clone, Debug, Copy)]
pub struct Vector3 {
	pub x: f32,
	pub y: f32,
	pub z: f32,
	pub _pad: f32
}

fn main() {
	ExampleBase::with_input_thread(
		[unsafe { NonZeroU32::new_unchecked(1200) }, unsafe { NonZeroU32::new_unchecked(675) }],
		|mut base| {
			let (renderpass, framebuffers) = unsafe {
				let renderpass_attachments = [
					vk::AttachmentDescription {
						format: base.present_image_views[0].format(),
						samples: vk::SampleCountFlags::TYPE_1,
						load_op: vk::AttachmentLoadOp::CLEAR,
						store_op: vk::AttachmentStoreOp::STORE,
						final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
						..Default::default()
					},
					vk::AttachmentDescription {
						format: vk::Format::D16_UNORM,
						samples: vk::SampleCountFlags::TYPE_1,
						load_op: vk::AttachmentLoadOp::CLEAR,
						initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
						final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
						..Default::default()
					}
				];
				let color_attachment_refs = [vk::AttachmentReference { attachment: 0, layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL }];
				let depth_attachment_ref = vk::AttachmentReference { attachment: 1, layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
				let dependencies = [vk::SubpassDependency {
					src_subpass: vk::SUBPASS_EXTERNAL,
					src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
					dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
					dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
					..Default::default()
				}];

				let subpasses = [vk::SubpassDescription::builder()
					.color_attachments(&color_attachment_refs)
					.depth_stencil_attachment(&depth_attachment_ref)
					.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
					.build()];

				let renderpass_create_info = vk::RenderPassCreateInfo::builder()
					.attachments(&renderpass_attachments)
					.subpasses(&subpasses)
					.dependencies(&dependencies);

				let renderpass = base
					.device
					.create_render_pass(&renderpass_create_info, None)
					.unwrap();

				let framebuffers: Vec<vk::Framebuffer> = base
					.present_image_views
					.iter()
					.map(|present_image_view| {
						let framebuffer_attachments = [*present_image_view.deref().deref(), *base.depth_image_view.deref().deref()];
						let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
							.render_pass(renderpass)
							.attachments(&framebuffer_attachments)
							.width(base.surface_size.width().get())
							.height(base.surface_size.height().get())
							.layers(1);

						base.device
							.create_framebuffer(&frame_buffer_create_info, None)
							.unwrap()
					})
					.collect();

				(renderpass, framebuffers)
			};

			let index_buffer_data = [0u32, 1, 2, 2, 3, 0];
			let index_buffer = {
				let buffer = Buffer::new(
					base.device.clone(),
					std::num::NonZeroU64::new(std::mem::size_of_val(&index_buffer_data) as u64).unwrap(),
					vk::BufferUsageFlags::INDEX_BUFFER,
					base.present_queue.deref().into(),
					BufferAllocatorParams::Some {
						allocator: &base.device_memory_allocator,
						requirements: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
					},
					Default::default()
				)
				.expect("Could not create index buffer");

				let memory = buffer.memory().unwrap();
				memory
					.map_memory_with(|mut access| {
						access.write_slice(
							&index_buffer_data,
							0,
							Default::default()
						);
						MappingAccessResult::Unmap
					})
					.expect("could not map memory");

				buffer
			};

			let vertices = [
				Vertex { pos: [-1.0, -1.0, 0.0, 1.0], uv: [0.0, 0.0] },
				Vertex { pos: [-1.0, 1.0, 0.0, 1.0], uv: [0.0, 1.0] },
				Vertex { pos: [1.0, 1.0, 0.0, 1.0], uv: [1.0, 1.0] },
				Vertex { pos: [1.0, -1.0, 0.0, 1.0], uv: [1.0, 0.0] }
			];
			let vertex_buffer = {
				let buffer = Buffer::new(
					base.device.clone(),
					std::num::NonZeroU64::new(std::mem::size_of_val(&vertices) as u64).unwrap(),
					vk::BufferUsageFlags::VERTEX_BUFFER,
					base.present_queue.deref().into(),
					BufferAllocatorParams::Some {
						allocator: &base.device_memory_allocator,
						requirements: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
					},
					Default::default()
				)
				.expect("Could not create index buffer");

				let memory = buffer.memory().unwrap();
				memory
					.map_memory_with(|mut access| {
						access.write_slice(&vertices, 0, Default::default());
						MappingAccessResult::Unmap
					})
					.expect("could not map memory");

				buffer
			};

			let uniform_color_buffer_data = Vector3 { x: 1.0, y: 1.0, z: 1.0, _pad: 0.0 };
			let uniform_color_buffer = {
				let buffer = Buffer::new(
					base.device.clone(),
					std::num::NonZeroU64::new(std::mem::size_of_val(&uniform_color_buffer_data) as u64).unwrap(),
					vk::BufferUsageFlags::UNIFORM_BUFFER,
					base.present_queue.deref().into(),
					BufferAllocatorParams::Some {
						allocator: &base.device_memory_allocator,
						requirements: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
					},
					Default::default()
				)
				.expect("could not create uniform color buffer");

				let memory = buffer.memory().unwrap();
				memory
					.map_memory_with(|mut access| {
						access.write_slice(
							&[uniform_color_buffer_data],
							0,
							Default::default()
						);
						MappingAccessResult::Unmap
					})
					.expect("could not map memory");

				buffer
			};

			let image = image::load_from_memory(include_bytes!("../../assets/rust.png"))
				.unwrap()
				.to_rgba8();
			let image_dimensions = image.dimensions();
			let image_data = image.into_raw();
			let image_buffer = {
				let buffer = Buffer::new(
					base.device.clone(),
					std::num::NonZeroU64::new((std::mem::size_of::<u8>() * image_data.len()) as u64).unwrap(),
					vk::BufferUsageFlags::TRANSFER_SRC,
					base.present_queue.deref().into(),
					BufferAllocatorParams::Some {
						allocator: &base.device_memory_allocator,
						requirements: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
					},
					Default::default()
				)
				.expect("could not create uniform color buffer");

				let memory = buffer.memory().unwrap();
				memory
					.map_memory_with(|mut access| {
						access.write_slice(&image_data, 0, Default::default());
						MappingAccessResult::Unmap
					})
					.expect("could not map memory");

				buffer
			};

			let texture_image_view = {
				let texture_image = Image::new(
					base.device.clone(),
					vk::Format::R8G8B8A8_UNORM,
					ImageSize::from(ImageSize::new_2d(
						NonZeroU32::new(image_dimensions.0).unwrap(),
						NonZeroU32::new(image_dimensions.1).unwrap(),
						NonZeroU32::new(1).unwrap(),
						MipmapLevels::One()
					))
					.into(),
					Default::default(),
					vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
					base.present_queue.deref().into(),
					ImageAllocatorParams::Some { allocator: &base.device_memory_allocator, requirements: vk::MemoryPropertyFlags::DEVICE_LOCAL },
					Default::default()
				)
				.expect("Could not create texture image");

				ImageView::new(
					texture_image.into(),
					ImageViewRange::Type2D(0, NonZeroU32::new(1).unwrap(), 0),
					None,
					Default::default(),
					vk::ImageAspectFlags::COLOR,
					Default::default()
				)
				.expect("Chould not create texture image view")
			};

			record_command_buffer(
				&base.setup_command_buffer,
				|command_buffer| {
					let device = command_buffer.pool().device();
					let cb_lock = command_buffer.lock().expect("vutex poisoned");

					let texture_barrier = vk::ImageMemoryBarrier {
						dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
						new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
						image: *texture_image_view.image().deref().deref(),
						subresource_range: vk::ImageSubresourceRange {
							aspect_mask: vk::ImageAspectFlags::COLOR,
							level_count: 1,
							layer_count: 1,
							..Default::default()
						},
						..Default::default()
					};
					unsafe {
						device.cmd_pipeline_barrier(
							*cb_lock,
							vk::PipelineStageFlags::BOTTOM_OF_PIPE,
							vk::PipelineStageFlags::TRANSFER,
							vk::DependencyFlags::empty(),
							&[],
							&[],
							&[texture_barrier]
						);
					}
					let buffer_copy_regions = vk::BufferImageCopy::builder()
						.image_subresource(
							vk::ImageSubresourceLayers::builder()
								.aspect_mask(vk::ImageAspectFlags::COLOR)
								.layer_count(1)
								.build()
						)
						.image_extent(vk::Extent3D { width: image_dimensions.0, height: image_dimensions.1, depth: 1 });

					unsafe {
						device.cmd_copy_buffer_to_image(
							*cb_lock,
							*image_buffer.deref().deref(),
							*texture_image_view.image().deref().deref(),
							vk::ImageLayout::TRANSFER_DST_OPTIMAL,
							&[buffer_copy_regions.build()]
						);
					}
					let texture_barrier_end = vk::ImageMemoryBarrier {
						src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
						dst_access_mask: vk::AccessFlags::SHADER_READ,
						old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
						new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
						image: *texture_image_view.image().deref().deref(),
						subresource_range: vk::ImageSubresourceRange {
							aspect_mask: vk::ImageAspectFlags::COLOR,
							level_count: 1,
							layer_count: 1,
							..Default::default()
						},
						..Default::default()
					};
					unsafe {
						device.cmd_pipeline_barrier(
							*cb_lock,
							vk::PipelineStageFlags::TRANSFER,
							vk::PipelineStageFlags::FRAGMENT_SHADER,
							vk::DependencyFlags::empty(),
							&[],
							&[],
							&[texture_barrier_end]
						);
					}
				}
			);
			submit_command_buffer_simple(
				&base.setup_command_buffer,
				&base.present_queue
			);

			let (
				viewports,
				scissors,
				pipeline_layout,
				desc_set_layouts,
				descriptor_pool,
				descriptor_sets,
				graphics_pipelines,
				vertex_shader_module,
				fragment_shader_module,
				sampler
			) = unsafe {
				let sampler_info = vk::SamplerCreateInfo {
					mag_filter: vk::Filter::LINEAR,
					min_filter: vk::Filter::LINEAR,
					mipmap_mode: vk::SamplerMipmapMode::LINEAR,
					address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
					address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
					address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
					max_anisotropy: 1.0,
					border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
					compare_op: vk::CompareOp::NEVER,
					..Default::default()
				};

				let sampler = base.device.create_sampler(&sampler_info, None).unwrap();

				let descriptor_sizes = [
					vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1 },
					vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 1 }
				];
				let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
					.pool_sizes(&descriptor_sizes)
					.max_sets(1);

				let descriptor_pool = base
					.device
					.create_descriptor_pool(&descriptor_pool_info, None)
					.unwrap();
				let desc_layout_bindings = [
					vk::DescriptorSetLayoutBinding {
						descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
						descriptor_count: 1,
						stage_flags: vk::ShaderStageFlags::FRAGMENT,
						..Default::default()
					},
					vk::DescriptorSetLayoutBinding {
						binding: 1,
						descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
						descriptor_count: 1,
						stage_flags: vk::ShaderStageFlags::FRAGMENT,
						..Default::default()
					}
				];
				let descriptor_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_layout_bindings);

				let desc_set_layouts = [base
					.device
					.create_descriptor_set_layout(&descriptor_info, None)
					.unwrap()];

				let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
					.descriptor_pool(descriptor_pool)
					.set_layouts(&desc_set_layouts);
				let descriptor_sets = base
					.device
					.allocate_descriptor_sets(&desc_alloc_info)
					.unwrap();

				let uniform_color_buffer_descriptor = vk::DescriptorBufferInfo {
					buffer: *uniform_color_buffer.deref().deref(),
					offset: 0,
					range: mem::size_of_val(&uniform_color_buffer_data) as u64
				};

				let tex_descriptor = vk::DescriptorImageInfo {
					image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
					image_view: *texture_image_view.deref().deref(),
					sampler
				};

				let write_desc_sets = [
					vk::WriteDescriptorSet {
						dst_set: descriptor_sets[0],
						descriptor_count: 1,
						descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
						p_buffer_info: &uniform_color_buffer_descriptor,
						..Default::default()
					},
					vk::WriteDescriptorSet {
						dst_set: descriptor_sets[0],
						dst_binding: 1,
						descriptor_count: 1,
						descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
						p_image_info: &tex_descriptor,
						..Default::default()
					}
				];
				base.device.update_descriptor_sets(&write_desc_sets, &[]);

				let mut vertex_spv_file = Cursor::new(&include_bytes!("../../shader/texture/vert.spv")[..]);
				let mut frag_spv_file = Cursor::new(&include_bytes!("../../shader/texture/frag.spv")[..]);

				let vertex_code = read_spv(&mut vertex_spv_file).expect("Failed to read vertex shader spv file");
				let vertex_shader_info = vk::ShaderModuleCreateInfo::builder().code(&vertex_code);

				let frag_code = read_spv(&mut frag_spv_file).expect("Failed to read fragment shader spv file");
				let frag_shader_info = vk::ShaderModuleCreateInfo::builder().code(&frag_code);

				let vertex_shader_module = base
					.device
					.create_shader_module(&vertex_shader_info, None)
					.expect("Vertex shader module error");

				let fragment_shader_module = base
					.device
					.create_shader_module(&frag_shader_info, None)
					.expect("Fragment shader module error");

				let layout_create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&desc_set_layouts);

				let pipeline_layout = base
					.device
					.create_pipeline_layout(&layout_create_info, None)
					.unwrap();

				let shader_entry_name = CString::new("main").unwrap();
				let shader_stage_create_infos = [
					vk::PipelineShaderStageCreateInfo {
						module: vertex_shader_module,
						p_name: shader_entry_name.as_ptr(),
						stage: vk::ShaderStageFlags::VERTEX,
						..Default::default()
					},
					vk::PipelineShaderStageCreateInfo {
						module: fragment_shader_module,
						p_name: shader_entry_name.as_ptr(),
						stage: vk::ShaderStageFlags::FRAGMENT,
						..Default::default()
					}
				];
				let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
					binding: 0,
					stride: mem::size_of::<Vertex>() as u32,
					input_rate: vk::VertexInputRate::VERTEX
				}];
				let vertex_input_attribute_descriptions = [
					vk::VertexInputAttributeDescription {
						location: 0,
						binding: 0,
						format: vk::Format::R32G32B32A32_SFLOAT,
						offset: Vertex::offsets().pos as u32
					},
					vk::VertexInputAttributeDescription {
						location: 1,
						binding: 0,
						format: vk::Format::R32G32_SFLOAT,
						offset: Vertex::offsets().uv as u32
					}
				];
				let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::builder()
					.vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
					.vertex_binding_descriptions(&vertex_input_binding_descriptions);

				let vertex_input_assembly_state_info =
					vk::PipelineInputAssemblyStateCreateInfo { topology: vk::PrimitiveTopology::TRIANGLE_LIST, ..Default::default() };
				let viewports = [vk::Viewport {
					x: 0.0,
					y: 0.0,
					width: base.surface_size.width().get() as f32,
					height: base.surface_size.height().get() as f32,
					min_depth: 0.0,
					max_depth: 1.0
				}];
				let scissors = [vk::Rect2D { extent: base.surface_size.into(), ..Default::default() }];
				let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
					.scissors(&scissors)
					.viewports(&viewports);

				let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
					front_face: vk::FrontFace::COUNTER_CLOCKWISE,
					line_width: 1.0,
					polygon_mode: vk::PolygonMode::FILL,
					..Default::default()
				};

				let multisample_state_info = vk::PipelineMultisampleStateCreateInfo::builder().rasterization_samples(vk::SampleCountFlags::TYPE_1);

				let noop_stencil_state = vk::StencilOpState {
					fail_op: vk::StencilOp::KEEP,
					pass_op: vk::StencilOp::KEEP,
					depth_fail_op: vk::StencilOp::KEEP,
					compare_op: vk::CompareOp::ALWAYS,
					..Default::default()
				};
				let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
					depth_test_enable: 1,
					depth_write_enable: 1,
					depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
					front: noop_stencil_state,
					back: noop_stencil_state,
					max_depth_bounds: 1.0,
					..Default::default()
				};

				let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
					blend_enable: 0,
					src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
					dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
					color_blend_op: vk::BlendOp::ADD,
					src_alpha_blend_factor: vk::BlendFactor::ZERO,
					dst_alpha_blend_factor: vk::BlendFactor::ZERO,
					alpha_blend_op: vk::BlendOp::ADD,
					color_write_mask: vk::ColorComponentFlags::RGBA
				}];
				let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
					.logic_op(vk::LogicOp::CLEAR)
					.attachments(&color_blend_attachment_states);

				let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
				let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

				let graphic_pipeline_infos = vk::GraphicsPipelineCreateInfo::builder()
					.stages(&shader_stage_create_infos)
					.vertex_input_state(&vertex_input_state_info)
					.input_assembly_state(&vertex_input_assembly_state_info)
					.viewport_state(&viewport_state_info)
					.rasterization_state(&rasterization_info)
					.multisample_state(&multisample_state_info)
					.depth_stencil_state(&depth_state_info)
					.color_blend_state(&color_blend_state)
					.dynamic_state(&dynamic_state_info)
					.layout(pipeline_layout)
					.render_pass(renderpass);

				let graphics_pipelines = base
					.device
					.create_graphics_pipelines(
						vk::PipelineCache::null(),
						&[graphic_pipeline_infos.build()],
						None
					)
					.unwrap();

				(
					viewports,
					scissors,
					pipeline_layout,
					desc_set_layouts,
					descriptor_pool,
					descriptor_sets,
					graphics_pipelines,
					vertex_shader_module,
					fragment_shader_module,
					sampler
				)
			};
			let graphics_pipeline = graphics_pipelines[0];

			base.render_loop(|base, present_index| {
				let clear_values = [
					vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] } },
					vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } }
				];

				let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
					.render_pass(renderpass)
					.framebuffer(framebuffers[present_index as usize])
					.render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: base.surface_size.into() })
					.clear_values(&clear_values);

				record_command_buffer(
					&base.draw_command_buffer,
					|command_buffer| {
						let device = command_buffer.pool().device();
						let cb_lock = command_buffer.lock().expect("vutex poisoned");

						unsafe {
							device.cmd_begin_render_pass(
								*cb_lock,
								&render_pass_begin_info,
								vk::SubpassContents::INLINE
							);
							device.cmd_bind_descriptor_sets(
								*cb_lock,
								vk::PipelineBindPoint::GRAPHICS,
								pipeline_layout,
								0,
								&descriptor_sets[..],
								&[]
							);
							device.cmd_bind_pipeline(
								*cb_lock,
								vk::PipelineBindPoint::GRAPHICS,
								graphics_pipeline
							);
							device.cmd_set_viewport(*cb_lock, 0, &viewports);
							device.cmd_set_scissor(*cb_lock, 0, &scissors);
							device.cmd_bind_vertex_buffers(
								*cb_lock,
								0,
								&[*vertex_buffer.deref().deref()],
								&[0]
							);
							device.cmd_bind_index_buffer(
								*cb_lock,
								*index_buffer.deref().deref(),
								0,
								vk::IndexType::UINT32
							);
							device.cmd_draw_indexed(
								*cb_lock,
								index_buffer_data.len() as u32,
								1,
								0,
								0,
								1
							);
							// Or draw without the index buffer
							// device.cmd_draw(draw_command_buffer, 3, 1, 0, 0);
							device.cmd_end_render_pass(*cb_lock);
						}
					}
				);
				submit_command_buffer(
					&base.draw_command_buffer,
					&base.present_queue,
					&base.present_complete_semaphore,
					vk::PipelineStageFlags::BOTTOM_OF_PIPE,
					&base.rendering_complete_semaphore
				);
			});
			unsafe {
				base.device.device_wait_idle().unwrap();
			}

			unsafe {
				for pipeline in graphics_pipelines {
					base.device.destroy_pipeline(pipeline, None);
				}
				base.device.destroy_pipeline_layout(pipeline_layout, None);
				base.device
					.destroy_shader_module(vertex_shader_module, None);
				base.device
					.destroy_shader_module(fragment_shader_module, None);

				for &descriptor_set_layout in desc_set_layouts.iter() {
					base.device
						.destroy_descriptor_set_layout(descriptor_set_layout, None);
				}
				base.device.destroy_descriptor_pool(descriptor_pool, None);
				base.device.destroy_sampler(sampler, None);
				for framebuffer in framebuffers {
					base.device.destroy_framebuffer(framebuffer, None);
				}
				base.device.destroy_render_pass(renderpass, None);
			}
		}
	);
}
