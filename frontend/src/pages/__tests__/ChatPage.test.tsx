import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../test/test-utils';
import ChatPage from '../ChatPage';

// =============================================================================
// Mocks
// =============================================================================

vi.mock('../../services/hooks/useChat', () => ({
  useChat: () => ({
    messages: [],
    isStreaming: false,
    currentResponse: '',
    sendMessage: vi.fn(),
    clearMessages: vi.fn(),
    suggestions: [],
  }),
}));

vi.mock('../../components/features/chat/MessageBubble', () => ({
  MessageBubble: ({ message }: { message: { id: string; content: string } }) => (
    <div data-testid={`message-${message.id}`}>{message.content}</div>
  ),
}));

vi.mock('../../components/features/chat/ChatInput', () => ({
  ChatInput: ({
    onSend,
    disabled,
    placeholder,
  }: {
    onSend: (msg: string) => void;
    disabled: boolean;
    placeholder: string;
  }) => (
    <div data-testid="chat-input">
      <input
        placeholder={placeholder}
        disabled={disabled}
        onChange={() => {}}
        data-testid="chat-input-field"
      />
      <button onClick={() => onSend('test')} data-testid="send-button">
        Send
      </button>
    </div>
  ),
}));

vi.mock('../../components/features/chat/SuggestionChips', () => ({
  SuggestionChips: ({
    suggestions,
    onSelect,
  }: {
    suggestions: string[];
    onSelect: (s: string) => void;
  }) => (
    <div data-testid="suggestion-chips">
      {suggestions.map((s) => (
        <button key={s} onClick={() => onSelect(s)} data-testid={`chip-${s}`}>
          {s}
        </button>
      ))}
    </div>
  ),
}));

vi.mock('../../components/features/chat/TypingIndicator', () => ({
  TypingIndicator: () => <div data-testid="typing-indicator">Typing...</div>,
}));

// =============================================================================
// Tests
// =============================================================================

describe('ChatPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  it('should render the chat interface header', () => {
    render(<ChatPage />);
    expect(
      screen.getByText('AI Chat Asszisztens'),
    ).toBeInTheDocument();
  });

  it('should render the welcome message when no messages exist', () => {
    render(<ChatPage />);
    expect(screen.getByText('Udvozollek!')).toBeInTheDocument();
  });

  it('should render the welcome description text', () => {
    render(<ChatPage />);
    expect(
      screen.getByText(/Kerdezz barmit a jarmuved problemjarol/),
    ).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Initial Suggestions
  // ---------------------------------------------------------------------------

  it('should show initial suggestion chips', () => {
    render(<ChatPage />);
    expect(screen.getByTestId('suggestion-chips')).toBeInTheDocument();
    expect(screen.getByText('Mi az a P0300 hibakod?')).toBeInTheDocument();
    expect(screen.getByText('Motor berreges okai')).toBeInTheDocument();
    expect(screen.getByText('Muszaki vizsgara keszulok')).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Chat Input
  // ---------------------------------------------------------------------------

  it('should render the chat input area', () => {
    render(<ChatPage />);
    expect(screen.getByTestId('chat-input')).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // No clear button when empty
  // ---------------------------------------------------------------------------

  it('should not show clear button when no messages exist', () => {
    render(<ChatPage />);
    expect(screen.queryByText('Torles')).not.toBeInTheDocument();
  });
});
