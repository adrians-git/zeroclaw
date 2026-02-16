use async_trait::async_trait;

/// An image attached to a channel message.
#[derive(Debug, Clone)]
pub struct ChannelImage {
    /// Raw image bytes.
    pub data: Vec<u8>,
    /// MIME type (e.g. "image/png").
    pub media_type: String,
}

/// A message received from or sent to a channel
#[derive(Debug, Clone)]
pub struct ChannelMessage {
    pub id: String,
    pub sender: String,
    pub content: String,
    pub channel: String,
    pub timestamp: u64,
    /// Images attached to this message (e.g. photos sent by the user).
    pub images: Vec<ChannelImage>,
}

/// Core channel trait — implement for any messaging platform
#[async_trait]
pub trait Channel: Send + Sync {
    /// Human-readable channel name
    fn name(&self) -> &str;

    /// Send a message through this channel
    async fn send(&self, message: &str, recipient: &str) -> anyhow::Result<()>;

    /// Start listening for incoming messages (long-running)
    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()>;

    /// Check if channel is healthy
    async fn health_check(&self) -> bool {
        true
    }

    /// Signal that the bot is processing a response (e.g. "typing" indicator).
    /// Implementations should repeat the indicator as needed for their platform.
    async fn start_typing(&self, _recipient: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Stop any active typing indicator.
    async fn stop_typing(&self, _recipient: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Send an image through this channel. Default is a no-op.
    async fn send_image(
        &self,
        _data: &[u8],
        _media_type: &str,
        _recipient: &str,
        _caption: Option<&str>,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}
