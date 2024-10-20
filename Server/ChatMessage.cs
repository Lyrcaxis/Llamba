public enum Role { System, Assistant, User }

public class ChatMessage {
    /// <summary> The role of the author of this message. Must be 'system', 'user', or 'assistant'. </summary>
    public string role { get; set; }

    /// <summary> The content of the message. </summary>
    public string content { get; set; }

    public ChatMessage() { }
    public ChatMessage(string message, Role role) => (this.content, this.role) = (message, role.ToString().ToLower());
}
