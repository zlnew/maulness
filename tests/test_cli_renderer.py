from unittest.mock import MagicMock, patch

from maulness.cli.renderer import TerminalLiveRenderer


def test_cli_renderer_empty_state():
    renderer = TerminalLiveRenderer(title="Test Empty")
    panel = renderer._render_view()
    assert panel.title == "Test Empty"


def test_cli_renderer_thoughts_and_tokens():
    with patch("maulness.cli.renderer.Live") as mock_live_cls:
        mock_live_instance = MagicMock()
        mock_live_cls.return_value = mock_live_instance

        renderer = TerminalLiveRenderer(title="Agent Run")
        with renderer:
            assert renderer._live is mock_live_instance
            mock_live_instance.__enter__.assert_called_once()

            # Append thought
            renderer.on_thought("Planning next move...")
            assert renderer._is_thinking is True
            assert len(renderer.thoughts) == 1
            mock_live_instance.update.assert_called()

            # Append token
            renderer.on_token("Here is the result.")
            assert renderer._is_thinking is False
            assert len(renderer.tokens) == 1
            mock_live_instance.update.assert_called()

        mock_live_instance.__exit__.assert_called_once()


def test_cli_renderer_only_tokens():
    renderer = TerminalLiveRenderer()
    renderer.on_token("Direct message without thought")
    panel = renderer._render_view()
    assert panel.title == "Agent Output"


def test_cli_renderer_exit_without_enter():
    renderer = TerminalLiveRenderer()
    # Should not raise
    renderer.__exit__(None, None, None)
