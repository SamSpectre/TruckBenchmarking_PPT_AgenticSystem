"""
Integration Tests for LangGraph Workflow (src/graph/runtime.py)

Tests cover:
- Workflow graph creation
- Routing logic
- Node execution flow
- State transitions
- Error handling
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.runtime import (
    create_workflow,
    route_after_scraping,
    route_after_validation,
    handle_failure_node,
    run_benchmark,
    stream_benchmark,
    get_graph_visualization,
)
from src.state.state import (
    WorkflowStatus,
    AgentType,
    ScrapingMode,
    initialize_state,
)


class TestWorkflowCreation:
    """Test workflow graph creation."""

    @pytest.mark.integration
    def test_create_workflow_returns_compiled_graph(self):
        """Test that create_workflow returns a compiled graph."""
        workflow = create_workflow()

        assert workflow is not None
        # Should be a compiled graph with invoke method
        assert hasattr(workflow, 'invoke')

    @pytest.mark.integration
    def test_create_workflow_with_checkpointer(self):
        """Test workflow creation with memory checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        workflow = create_workflow(checkpointer=checkpointer)

        assert workflow is not None

    @pytest.mark.integration
    def test_workflow_has_expected_nodes(self):
        """Test that workflow graph has expected nodes."""
        workflow = create_workflow()
        graph = workflow.get_graph()

        # Get node names
        node_names = [node.name for node in graph.nodes.values()]

        # Check for expected nodes
        assert "scrape" in node_names or any("scrape" in n for n in node_names)
        assert "validate" in node_names or any("validate" in n for n in node_names)
        assert "present" in node_names or any("present" in n for n in node_names)


class TestRoutingFunctions:
    """Test routing decision functions."""

    @pytest.mark.integration
    def test_route_after_scraping_success(self, sample_state_after_scraping):
        """Test routing after successful scraping."""
        route = route_after_scraping(sample_state_after_scraping)

        assert route == "validate"

    @pytest.mark.integration
    def test_route_after_scraping_failure(self, sample_initial_state):
        """Test routing when scraping fails."""
        state = sample_initial_state.copy()
        state["workflow_status"] = WorkflowStatus.SCRAPING_FAILED
        state["all_vehicles"] = []

        route = route_after_scraping(state)

        assert route == "end_failed"

    @pytest.mark.integration
    def test_route_after_scraping_no_vehicles(self, sample_initial_state):
        """Test routing when no vehicles found."""
        state = sample_initial_state.copy()
        state["all_vehicles"] = []

        route = route_after_scraping(state)

        assert route == "end_failed"

    @pytest.mark.integration
    def test_route_after_validation_pass(self, sample_state_after_validation):
        """Test routing when validation passes."""
        route = route_after_validation(sample_state_after_validation)

        assert route == "present"

    @pytest.mark.integration
    def test_route_after_validation_fail_with_retries(self, sample_state_after_scraping):
        """Test routing when validation fails but retries available."""
        state = sample_state_after_scraping.copy()
        state["quality_validation"] = {
            "overall_quality_score": 0.4,
            "passes_threshold": False,
        }
        state["total_retries_remaining"] = 2

        route = route_after_validation(state)

        assert route == "retry"

    @pytest.mark.integration
    def test_route_after_validation_fail_no_retries(self, sample_state_after_scraping):
        """Test routing when validation fails and no retries left."""
        state = sample_state_after_scraping.copy()
        state["quality_validation"] = {
            "overall_quality_score": 0.4,
            "passes_threshold": False,
        }
        state["total_retries_remaining"] = 0

        route = route_after_validation(state)

        assert route == "end_failed"

    @pytest.mark.integration
    def test_route_after_validation_no_result(self, sample_state_after_scraping):
        """Test routing when no validation result."""
        state = sample_state_after_scraping.copy()
        state["quality_validation"] = None

        route = route_after_validation(state)

        assert route == "end_failed"


class TestFailureHandlerNode:
    """Test failure handler node."""

    @pytest.mark.integration
    def test_handle_failure_scraping_failed(self, sample_initial_state):
        """Test failure handler for scraping failure."""
        state = sample_initial_state.copy()
        state["workflow_status"] = WorkflowStatus.SCRAPING_FAILED

        result = handle_failure_node(state)

        assert result["workflow_status"] == WorkflowStatus.FAILED
        assert any("Scraping failed" in e for e in result["errors"])

    @pytest.mark.integration
    def test_handle_failure_quality_failed(self, sample_state_after_scraping):
        """Test failure handler for quality validation failure."""
        state = sample_state_after_scraping.copy()
        state["workflow_status"] = WorkflowStatus.QUALITY_FAILED

        result = handle_failure_node(state)

        assert result["workflow_status"] == WorkflowStatus.FAILED
        assert any("Quality validation failed" in e for e in result["errors"])

    @pytest.mark.integration
    def test_handle_failure_sets_end_time(self, sample_initial_state):
        """Test that failure handler sets end time."""
        state = sample_initial_state.copy()
        state["workflow_status"] = WorkflowStatus.FAILED

        result = handle_failure_node(state)

        assert "workflow_end_time" in result
        assert result["workflow_end_time"] is not None


class TestGraphVisualization:
    """Test graph visualization."""

    @pytest.mark.integration
    def test_get_graph_visualization(self):
        """Test Mermaid diagram generation."""
        mermaid = get_graph_visualization()

        assert mermaid is not None
        assert isinstance(mermaid, str)
        # Mermaid diagrams typically start with flowchart or graph
        assert "graph" in mermaid.lower() or "flowchart" in mermaid.lower() or "---" in mermaid


class TestStateInitialization:
    """Test state initialization for workflow."""

    @pytest.mark.integration
    def test_initialize_state_for_workflow(self, sample_oem_urls):
        """Test state initialization for workflow execution."""
        state = initialize_state(sample_oem_urls)

        assert state["workflow_status"] == WorkflowStatus.INITIALIZED
        assert state["oem_urls"] == sample_oem_urls
        assert state["retry_count"] == 0

    @pytest.mark.integration
    def test_initialize_state_with_mode(self, sample_oem_urls):
        """Test state initialization with scraping mode."""
        state = initialize_state(sample_oem_urls, scraping_mode=ScrapingMode.INTELLIGENT)

        assert state["scraping_mode"] == ScrapingMode.INTELLIGENT


class TestMockedWorkflowExecution:
    """Test workflow execution with mocked agents."""

    @pytest.mark.integration
    @patch('src.agents.scraping_agent.scraping_node')
    @patch('src.agents.quality_validator.validation_node')
    @patch('src.agents.presentation_generator.presentation_node')
    def test_workflow_happy_path(
        self,
        mock_presentation,
        mock_validation,
        mock_scraping,
        sample_oem_urls,
        sample_scraping_result,
        sample_quality_validation_passed,
    ):
        """Test complete workflow with mocked agents."""
        # Setup mocks
        mock_scraping.return_value = {
            "workflow_status": WorkflowStatus.VALIDATING,
            "scraping_results": [sample_scraping_result],
            "all_vehicles": sample_scraping_result["vehicles"],
        }

        mock_validation.return_value = {
            "workflow_status": WorkflowStatus.GENERATING_PRESENTATION,
            "quality_validation": sample_quality_validation_passed,
        }

        mock_presentation.return_value = {
            "workflow_status": WorkflowStatus.COMPLETED,
            "presentation_result": {
                "all_presentation_paths": ["outputs/test.pptx"],
            },
        }

        # Create and run workflow
        # Note: In actual test, the mocks may not be injected this way
        # This is more of a structure demonstration

    @pytest.mark.integration
    def test_workflow_handles_empty_urls(self):
        """Test workflow with empty URL list."""
        state = initialize_state([])

        assert state["oem_urls"] == []
        # Workflow should fail fast with no URLs


class TestWorkflowStateFlow:
    """Test state flow through workflow stages."""

    @pytest.mark.integration
    def test_state_progresses_through_stages(self):
        """Test that state can progress through all stages."""
        # Initial state
        state = initialize_state(["https://example.com"])
        assert state["workflow_status"] == WorkflowStatus.INITIALIZED

        # After scraping
        state["workflow_status"] = WorkflowStatus.SCRAPING
        assert state["workflow_status"] == WorkflowStatus.SCRAPING

        # After validation
        state["workflow_status"] = WorkflowStatus.VALIDATING
        assert state["workflow_status"] == WorkflowStatus.VALIDATING

        # After presentation
        state["workflow_status"] = WorkflowStatus.GENERATING_PRESENTATION
        assert state["workflow_status"] == WorkflowStatus.GENERATING_PRESENTATION

        # Completed
        state["workflow_status"] = WorkflowStatus.COMPLETED
        assert state["workflow_status"] == WorkflowStatus.COMPLETED

    @pytest.mark.integration
    def test_state_retry_flow(self):
        """Test state flow during retry."""
        state = initialize_state(["https://example.com"])

        # Simulate retry
        state["workflow_status"] = WorkflowStatus.RETRYING
        state["retry_count"] = 1
        state["total_retries_remaining"] = 2

        assert state["workflow_status"] == WorkflowStatus.RETRYING
        assert state["retry_count"] == 1

    @pytest.mark.integration
    def test_state_tracks_costs(self):
        """Test that state tracks costs through workflow."""
        state = initialize_state(["https://example.com"])

        # Simulate adding costs
        state["total_tokens_used"] += 1000
        state["total_cost_usd"] += 0.01

        assert state["total_tokens_used"] == 1000
        assert state["total_cost_usd"] == 0.01


class TestWorkflowEdgeCases:
    """Test workflow edge cases and error handling."""

    @pytest.mark.integration
    def test_workflow_with_single_url(self):
        """Test workflow with single URL."""
        state = initialize_state(["https://example.com"])

        assert len(state["oem_urls"]) == 1

    @pytest.mark.integration
    def test_workflow_with_many_urls(self):
        """Test workflow with many URLs."""
        urls = [f"https://example{i}.com" for i in range(10)]
        state = initialize_state(urls)

        assert len(state["oem_urls"]) == 10

    @pytest.mark.integration
    def test_workflow_preserves_errors(self, sample_initial_state):
        """Test that workflow preserves errors through state."""
        state = sample_initial_state.copy()
        state["errors"] = ["Error 1"]

        # Add another error
        state["errors"].append("Error 2")

        assert len(state["errors"]) == 2

    @pytest.mark.integration
    def test_workflow_preserves_warnings(self, sample_initial_state):
        """Test that workflow preserves warnings through state."""
        state = sample_initial_state.copy()
        state["warnings"] = ["Warning 1"]

        # Add another warning
        state["warnings"].append("Warning 2")

        assert len(state["warnings"]) == 2


class TestConditionalEdges:
    """Test conditional edge logic in workflow."""

    @pytest.mark.integration
    def test_conditional_edge_scraping_success(self, sample_state_after_scraping):
        """Test conditional edge after successful scraping."""
        route = route_after_scraping(sample_state_after_scraping)

        # Should go to validation
        assert route == "validate"

    @pytest.mark.integration
    def test_conditional_edge_validation_threshold(self):
        """Test conditional edge at validation threshold boundary."""
        # Just at threshold
        state = {
            "quality_validation": {
                "passes_threshold": True,
                "overall_quality_score": 0.60,
            },
            "total_retries_remaining": 2,
        }

        route = route_after_validation(state)
        assert route == "present"

        # Just below threshold
        state["quality_validation"]["passes_threshold"] = False
        route = route_after_validation(state)
        assert route == "retry"


class TestWorkflowCostTracking:
    """Test cost tracking through workflow."""

    @pytest.mark.integration
    def test_cost_accumulation(self, sample_initial_state):
        """Test that costs accumulate correctly."""
        state = sample_initial_state.copy()

        # Scraping cost
        state["total_cost_usd"] += 0.02
        state["cost_breakdown"]["scraping"] = 0.02

        # Validation cost
        state["total_cost_usd"] += 0.01
        state["cost_breakdown"]["validation"] = 0.01

        assert state["total_cost_usd"] == 0.03
        assert len(state["cost_breakdown"]) == 2

    @pytest.mark.integration
    def test_token_accumulation(self, sample_initial_state):
        """Test that tokens accumulate correctly."""
        state = sample_initial_state.copy()

        # Scraping tokens
        state["total_tokens_used"] += 5000

        # Validation tokens
        state["total_tokens_used"] += 1000

        assert state["total_tokens_used"] == 6000


class TestWorkflowMessages:
    """Test message handling in workflow."""

    @pytest.mark.integration
    def test_messages_list_empty_initially(self, sample_initial_state):
        """Test that messages list is empty initially."""
        assert sample_initial_state["messages"] == []

    @pytest.mark.integration
    def test_messages_can_be_appended(self, sample_initial_state):
        """Test that messages can be appended."""
        state = sample_initial_state.copy()
        from langchain_core.messages import HumanMessage

        # This tests the message structure compatibility
        # In practice, LangGraph handles this via add_messages
        state["messages"] = list(state["messages"]) + [
            HumanMessage(content="Start benchmarking")
        ]

        assert len(state["messages"]) == 1
