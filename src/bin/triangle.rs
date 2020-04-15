use std::{default::Default, ffi::CString, io::Cursor, mem, num::NonZeroU32, ops::Deref};

use vulkayes_core::ash::{util::*, version::DeviceV1_0, vk};

use examples::state::ApplicationState;
use vulkayes_core::{
	memory::device::{naive::NaiveDeviceMemoryAllocator, MappingAccessResult},
	pipeline::layout::PipelineLayout,
	prelude::*,
	resource::buffer
};

vulkayes_core::offsetable_struct! {
	#[derive(Copy, Clone, Debug)]
	struct Vertex {
		pos: [f32; 4],
		color: [f32; 4]
	} repr(C) as VertexOffsets
}

struct TriangleState {
	vertex_shader_module: vk::ShaderModule,
	fragment_shader_module: vk::ShaderModule,

	#[allow(dead_code)] // keep alive
	index_buffer: Vrc<Buffer>,
	#[allow(dead_code)] // keep alive
	vertex_buffer: Vrc<Buffer>,

	framebuffers: Vec<vk::Framebuffer>,
	scissors: vk::Rect2D,
	viewport: vk::Viewport,

	render_pass: vk::RenderPass,

	#[allow(dead_code)] // keep alive
	pipeline_layout: Vrc<PipelineLayout>,
	graphic_pipelines: Vec<vk::Pipeline>
}
impl TriangleState {
	fn create_render_pass(device: Vrc<Device>, format: vk::Format) -> vk::RenderPass {
		let render_pass_attachments = [
			vk::AttachmentDescription {
				format,
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
		let color_attachment_refs = [vk::AttachmentReference {
			attachment: 0,
			layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
		}];
		let depth_attachment_ref = vk::AttachmentReference {
			attachment: 1,
			layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		};
		let dependencies = [vk::SubpassDependency {
			src_subpass: vk::SUBPASS_EXTERNAL,
			src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
			dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
				| vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
			dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
			..Default::default()
		}];

		let subpasses = [vk::SubpassDescription::builder()
			.color_attachments(&color_attachment_refs)
			.depth_stencil_attachment(&depth_attachment_ref)
			.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
			.build()];

		let render_pass_create_info = vk::RenderPassCreateInfo::builder()
			.attachments(&render_pass_attachments)
			.subpasses(&subpasses)
			.dependencies(&dependencies);

		let render_pass = unsafe {
			device
				.create_render_pass(&render_pass_create_info, None)
				.unwrap()
		};

		render_pass
	}

	fn create_framebuffers<'a>(
		renderpass: vk::RenderPass,
		present_image_views: impl Iterator<Item = &'a Vrc<ImageView>>,
		depth_image_view: &Vrc<ImageView>
	) -> Vec<vk::Framebuffer> {
		let framebuffers: Vec<vk::Framebuffer> = present_image_views
			.map(|present_image_view| {
				let framebuffer_attachments = [
					*present_image_view.deref().deref(),
					*depth_image_view.deref().deref()
				];
				let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
					.render_pass(renderpass)
					.attachments(&framebuffer_attachments)
					.width(present_image_view.subresource_image_size().width().get())
					.height(present_image_view.subresource_image_size().height().get())
					.layers(1);

				vulkayes_core::log::debug!(
					"Creating framebuffers for {:#?}",
					framebuffer_attachments
				);
				unsafe {
					present_image_view
						.image()
						.device()
						.create_framebuffer(&frame_buffer_create_info, None)
						.unwrap()
				}
			})
			.collect();

		framebuffers
	}

	fn create_buffers(
		present_queue: &Queue,
		device_memory_allocator: &NaiveDeviceMemoryAllocator
	) -> (Vrc<Buffer>, Vrc<Buffer>) {
		let index_buffer = {
			let index_buffer_data = [0u32, 1, 2];

			let buffer = Buffer::new(
				present_queue.device().clone(),
				std::num::NonZeroU64::new(std::mem::size_of_val(&index_buffer_data) as u64)
					.unwrap(),
				vk::BufferUsageFlags::INDEX_BUFFER,
				present_queue.into(),
				buffer::params::AllocatorParams::Some {
					allocator: device_memory_allocator,
					requirements: vk::MemoryPropertyFlags::HOST_VISIBLE
						| vk::MemoryPropertyFlags::HOST_COHERENT
				},
				Default::default()
			)
			.expect("Could not create index buffer");

			let memory = buffer.memory().unwrap();
			memory
				.map_memory_with(|mut access| {
					access.write_slice(&index_buffer_data, 0, Default::default());
					MappingAccessResult::Unmap
				})
				.expect("could not map memory");

			buffer
		};

		let vertex_buffer = {
			let buffer = Buffer::new(
				present_queue.device().clone(),
				std::num::NonZeroU64::new(3 * std::mem::size_of::<Vertex>() as u64).unwrap(),
				vk::BufferUsageFlags::VERTEX_BUFFER,
				present_queue.into(),
				buffer::params::AllocatorParams::Some {
					allocator: device_memory_allocator,
					requirements: vk::MemoryPropertyFlags::HOST_VISIBLE
						| vk::MemoryPropertyFlags::HOST_COHERENT
				},
				Default::default()
			)
			.expect("Could not create index buffer");

			let vertices = [
				Vertex {
					pos: [-1.0, 1.0, 0.0, 1.0],
					color: [0.0, 1.0, 0.0, 1.0]
				},
				Vertex {
					pos: [1.0, 1.0, 0.0, 1.0],
					color: [0.0, 0.0, 1.0, 1.0]
				},
				Vertex {
					pos: [0.0, -1.0, 0.0, 1.0],
					color: [1.0, 0.0, 0.0, 1.0]
				}
			];

			let memory = buffer.memory().unwrap();
			memory
				.map_memory_with(|mut access| {
					access.write_slice(&vertices, 0, Default::default());
					MappingAccessResult::Unmap
				})
				.expect("could not map memory");

			buffer
		};

		(index_buffer, vertex_buffer)
	}

	fn create_shaders(device: &Device) -> (vk::ShaderModule, vk::ShaderModule) {
		let mut vertex_spv_file =
			Cursor::new(&include_bytes!("../../shader/triangle/vert.spv")[..]);
		let mut frag_spv_file = Cursor::new(&include_bytes!("../../shader/triangle/frag.spv")[..]);

		let vertex_code =
			read_spv(&mut vertex_spv_file).expect("Failed to read vertex shader spv file");
		let vertex_shader_info = vk::ShaderModuleCreateInfo::builder().code(&vertex_code);

		let frag_code =
			read_spv(&mut frag_spv_file).expect("Failed to read fragment shader spv file");
		let frag_shader_info = vk::ShaderModuleCreateInfo::builder().code(&frag_code);

		let vertex_shader_module = unsafe {
			device
				.create_shader_module(&vertex_shader_info, None)
				.expect("Vertex shader module error")
		};

		let fragment_shader_module = unsafe {
			device
				.create_shader_module(&frag_shader_info, None)
				.expect("Fragment shader module error")
		};

		(vertex_shader_module, fragment_shader_module)
	}

	fn create_pipeline(
		vertex_shader_module: vk::ShaderModule,
		fragment_shader_module: vk::ShaderModule,
		scissors: vk::Rect2D,
		viewport: vk::Viewport,
		pipeline_layout: &PipelineLayout,
		renderpass: vk::RenderPass
	) -> Vec<vk::Pipeline> {
		let shader_entry_name = CString::new("main").unwrap();
		let shader_stage_create_infos = [
			vk::PipelineShaderStageCreateInfo {
				module: vertex_shader_module,
				p_name: shader_entry_name.as_ptr(),
				stage: vk::ShaderStageFlags::VERTEX,
				..Default::default()
			},
			vk::PipelineShaderStageCreateInfo {
				s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
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
				format: vk::Format::R32G32B32A32_SFLOAT,
				offset: Vertex::offsets().color as u32
			}
		];

		let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
			vertex_attribute_description_count: vertex_input_attribute_descriptions.len() as u32,
			p_vertex_attribute_descriptions: vertex_input_attribute_descriptions.as_ptr(),
			vertex_binding_description_count: vertex_input_binding_descriptions.len() as u32,
			p_vertex_binding_descriptions: vertex_input_binding_descriptions.as_ptr(),
			..Default::default()
		};
		let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
			topology: vk::PrimitiveTopology::TRIANGLE_LIST,
			..Default::default()
		};

		let scissors = [scissors];
		let viewports = [viewport];
		let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
			.scissors(&scissors)
			.viewports(&viewports);

		let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
			front_face: vk::FrontFace::COUNTER_CLOCKWISE,
			line_width: 1.0,
			polygon_mode: vk::PolygonMode::FILL,
			..Default::default()
		};
		let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
			rasterization_samples: vk::SampleCountFlags::TYPE_1,
			..Default::default()
		};
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
			color_write_mask: vk::ColorComponentFlags::all()
		}];
		let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
			.logic_op(vk::LogicOp::CLEAR)
			.attachments(&color_blend_attachment_states);

		let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
		let dynamic_state_info =
			vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

		let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
			.stages(&shader_stage_create_infos)
			.vertex_input_state(&vertex_input_state_info)
			.input_assembly_state(&vertex_input_assembly_state_info)
			.viewport_state(&viewport_state_info)
			.rasterization_state(&rasterization_info)
			.multisample_state(&multisample_state_info)
			.depth_stencil_state(&depth_state_info)
			.color_blend_state(&color_blend_state)
			.dynamic_state(&dynamic_state_info)
			.layout(*pipeline_layout.deref())
			.render_pass(renderpass);

		let graphics_pipelines = unsafe {
			pipeline_layout
				.device()
				.create_graphics_pipelines(
					vk::PipelineCache::null(),
					&[graphic_pipeline_info.build()],
					None
				)
				.expect("Unable to create graphics pipeline")
		};

		graphics_pipelines
	}

	fn new(state: &mut ApplicationState) -> Self {
		let render_pass = Self::create_render_pass(
			state.base.device.clone(),
			state.swapchain.present_views[0].format()
		);

		let framebuffers = Self::create_framebuffers(
			render_pass,
			state.swapchain.present_views.iter(),
			&state.depth_image_view
		);

		let (index_buffer, vertex_buffer) =
			Self::create_buffers(&state.base.present_queue, &state.device_memory_allocator);

		let (vertex_shader_module, fragment_shader_module) =
			Self::create_shaders(&state.base.device);

		let pipeline_layout = PipelineLayout::new(
			state.base.device.clone(),
			std::iter::empty(),
			std::iter::empty(),
			Default::default()
		)
		.expect("could not create pipeline layout");
		let scissors = vk::Rect2D {
			offset: vk::Offset2D { x: 0, y: 0 },
			extent: vk::Extent2D {
				width: state.surface_size[0].get(),
				height: state.surface_size[1].get()
			}
		};
		let viewport = vk::Viewport {
			x: 0.0,
			y: 0.0,
			width: state.surface_size[0].get() as f32,
			height: state.surface_size[1].get() as f32,
			min_depth: 0.0,
			max_depth: 1.0
		};

		let graphic_pipelines = Self::create_pipeline(
			vertex_shader_module,
			fragment_shader_module,
			scissors,
			viewport,
			&pipeline_layout,
			render_pass
		);

		TriangleState {
			vertex_shader_module,
			fragment_shader_module,

			index_buffer,
			vertex_buffer,

			framebuffers,
			scissors,
			viewport,

			render_pass,

			pipeline_layout,
			graphic_pipelines
		}
	}
}

fn main() {
	examples::state::ApplicationState::new_input_thread(
		unsafe {
			[
				NonZeroU32::new_unchecked(1200),
				NonZeroU32::new_unchecked(675)
			]
		},
		|state| TriangleState::new(state),
		|state, me| {
			let next_frame = state.acquire_next_frame();

			if next_frame.state().swapchain_recreated() {
				me.scissors = vk::Rect2D {
					offset: vk::Offset2D { x: 0, y: 0 },
					extent: vk::Extent2D {
						width: next_frame.state().surface_size[0].get(),
						height: next_frame.state().surface_size[1].get()
					}
				};
				me.viewport = vk::Viewport {
					x: 0.0,
					y: 0.0,
					width: next_frame.state().surface_size[0].get() as f32,
					height: next_frame.state().surface_size[1].get() as f32,
					min_depth: 0.0,
					max_depth: 1.0
				};

				me.framebuffers = TriangleState::create_framebuffers(
					me.render_pass,
					next_frame.state().swapchain.present_views.iter(),
					&next_frame.state().depth_image_view
				);
			}

			let clear_values = [
				vk::ClearValue {
					color: vk::ClearColorValue {
						float32: [0.0, 0.0, 0.0, 0.0]
					}
				},
				vk::ClearValue {
					depth_stencil: vk::ClearDepthStencilValue {
						depth: 1.0,
						stencil: 0
					}
				}
			];

			let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
				.render_pass(me.render_pass)
				.framebuffer(me.framebuffers[next_frame.present_index() as usize])
				.render_area(me.scissors) // technically the same thing
				.clear_values(&clear_values);

			unsafe {
				let device = next_frame.command_buffer().pool().device();
				let cb_lock = next_frame.command_buffer().lock().expect("vutex poisoned");

				device.cmd_begin_render_pass(
					*cb_lock,
					&render_pass_begin_info,
					vk::SubpassContents::INLINE
				);
				device.cmd_bind_pipeline(
					*cb_lock,
					vk::PipelineBindPoint::GRAPHICS,
					me.graphic_pipelines[0]
				);
				device.cmd_set_viewport(*cb_lock, 0, &[me.viewport]);
				device.cmd_set_scissor(*cb_lock, 0, &[me.scissors]);

				device.cmd_bind_vertex_buffers(
					*cb_lock,
					0,
					&[*me.vertex_buffer.deref().deref()],
					&[0]
				);
				device.cmd_bind_index_buffer(
					*cb_lock,
					*me.index_buffer.deref().deref(),
					0,
					vk::IndexType::UINT32
				);
				device.cmd_draw_indexed(
					*cb_lock,
					me.index_buffer.size_of::<u32>() as u32,
					1,
					0,
					0,
					1
				);
				// Or draw without the index buffer
				// device.cmd_draw(draw_command_buffer, 3, 1, 0, 0);

				device.cmd_end_render_pass(*cb_lock);
			}

			next_frame.submit_and_present();
		},
		|state, me| {
			unsafe {
				state.base.device.device_wait_idle().unwrap();
				for pipeline in me.graphic_pipelines {
					state.base.device.destroy_pipeline(pipeline, None);
				}

				state
					.base
					.device
					.destroy_shader_module(me.vertex_shader_module, None);
				state
					.base
					.device
					.destroy_shader_module(me.fragment_shader_module, None);

				for framebuffer in me.framebuffers {
					state.base.device.destroy_framebuffer(framebuffer, None);
				}

				state.base.device.destroy_render_pass(me.render_pass, None);

				// rest is dropped implicitly afterwards
			}
		}
	)
}
