use std::sync::mpsc::{RecvError, SendError, SyncSender};
use std::thread::JoinHandle;

pub enum Action<T> {
    Process(T),
    Finish,
}

pub enum ThreadError<I> {
    InnerFn(I),
    Recv(RecvError),
}

impl<I> std::fmt::Display for ThreadError<I>
where
    I: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThreadError::InnerFn(inner) => write!(f, "Processing Failed: {inner}"),
            ThreadError::Recv(err) => write!(f, "Could not receive from channel: {err}"),
        }
    }
}

#[derive(Clone)]
pub struct ExecuteSender<R> {
    pub tx: SyncSender<Action<R>>,
}

impl<R> ExecuteSender<R> {
    pub fn create<F, I>(f: F) -> (Self, JoinHandle<Result<(), ThreadError<I>>>)
    where
        R: Send + 'static,
        I: Send + std::fmt::Display + 'static,
        F: Fn(R) -> Result<(), I> + Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let handle = std::thread::spawn(move || loop {
            match rx.recv() {
                Ok(input_val) => match input_val {
                    Action::Finish => return Ok(()),
                    Action::Process(v) => f(v).map_err(|err| ThreadError::InnerFn(err))?,
                },
                Err(err) => return Err(ThreadError::Recv(err)),
            }
        });

        (ExecuteSender { tx }, handle)
    }

    pub fn queue(&self, v: R) -> Result<(), SendError<Action<R>>> {
        self.tx.send(Action::Process(v))
    }

    pub fn finish(self) -> Result<(), SendError<Action<R>>> {
        self.tx.send(Action::Finish)
    }
}
