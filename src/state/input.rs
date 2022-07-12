use std::{num::NonZeroU32, sync::mpsc};

use vulkayes_core::log;
use vulkayes_window::winit::winit;
use winit::window::Window;

#[derive(Debug)]
pub enum InputEvent {
	Exit,
	WindowSize([NonZeroU32; 2])
}

impl super::ApplicationState {
	/// Creates new `ApplicationState`. Uses the current thread as input thread and starts a new render thread.
	/// Also initializes the logger and window.
	///
	/// On certain systems, such as macOS, it is required to run the input thread on the main thread. This function
	/// thus has to be called from the main thread on such systems.
	pub(super) fn new_input_thread_inner(
		window_size: [NonZeroU32; 2],
		render_thread_fn: impl FnOnce(Window, mpsc::Receiver<InputEvent>, mpsc::Sender<Window>) + Send + 'static
	) -> ! {
		use edwardium_logger::{
			targets::{stderr::StderrTarget, util::ignore_list::IgnoreList},
			Logger
		};

		// Register a logger since Vulkayes logs through the log crate
		let logger = Logger::new(
			StderrTarget::new(
				log::Level::Trace,
				IgnoreList::EMPTY_PATTERNS
			),
			std::time::Instant::now()
		);
		logger.init_boxed().expect("Could not initialize logger");

		let event_loop = winit::event_loop::EventLoop::new();
		let window = winit::window::WindowBuilder::new()
			.with_title("Vulkayes - Example")
			.with_inner_size(winit::dpi::LogicalSize::new(
				f64::from(window_size[0].get()),
				f64::from(window_size[1].get())
			))
			.build(&event_loop)
			.expect("could not create window");

		// A little something about macOS and winit
		// On macOS, both the even loop and the window must stay in the main thread while rendering.
		// However, the window is needed to create surface, which needs an instance, which means it needs to happen in the renderer thread.
		// Here we have to move the window into the new thread and then send it back using a channel.
		let (oneoff_sender, oneoff_receiver) = mpsc::channel::<Window>();

		let (input_sender, input_receiver) = mpsc::channel::<InputEvent>();
		let render_thread = std::thread::spawn(move || {
			render_thread_fn(window, input_receiver, oneoff_sender);
		});

		// We don't actually need the window after, but it has to be in this thread. Weird.
		let window = oneoff_receiver
			.recv()
			.expect("could not receive window back");

		Self::input_loop(
			input_sender,
			event_loop,
			window,
			render_thread
		)
	}

	fn input_loop(
		input_sender: mpsc::Sender<InputEvent>,
		event_loop: winit::event_loop::EventLoop<()>,
		_window: Window,
		render_thread: std::thread::JoinHandle<()>
	) -> ! {
		use winit::{
			event::{Event, WindowEvent},
			event_loop::ControlFlow
		};

		// To join the render thread after close has been requested, we need access to owned value of `render_thread` inside the event handler closure below.
		// Wrapping it in `Option` allows us to take it out once we know the application is definitely closing, which doesn't have to be the last time the closure
		// is invoked, however.
		let mut render_thread_movable = Some(render_thread);

		// Technically result is now `!`, but this can be trivially replaced by `run_return` where supported
		event_loop.run(move |event, _target, control_flow| {
			// Handle window close event
			let event_to_send = match event {
				Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
					*control_flow = ControlFlow::Exit;
					Some(InputEvent::Exit)
				}
				Event::WindowEvent { event: WindowEvent::Resized(size), .. } => match (
					NonZeroU32::new(size.width),
					NonZeroU32::new(size.height)
				) {
					(Some(w), Some(h)) => Some(InputEvent::WindowSize([w, h])),
					_ => None
				},
				_ => None
			};

			if let Some(event) = event_to_send {
				input_sender
					.send(event)
					.expect("could not send input event");
			}

			if *control_flow == ControlFlow::Exit {
				if let Some(render_thread) = render_thread_movable.take() {
					render_thread.join().expect("Could not join render thread");
				}
			}
		})
	}
}
