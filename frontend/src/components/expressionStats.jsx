import Plot from "react-plotly.js";
import SectionTitle from "./sectionTitle";

const ExpressionStats = ({ expressionStats, title }) => {
  const emotions = Object.keys(expressionStats || {}).map(
    (emotion) => emotion.charAt(0).toUpperCase() + emotion.slice(1),
  );
  const scores = Object.values(expressionStats || {}).map((score) =>
    typeof score === "string" ? parseFloat(score) : score,
  );

  // Create the chart data
  const data = [
    {
      x: emotions,
      y: scores,
      type: "bar",
      marker: {
        color: [
          "#06B6D4",
          "#10B981",
          "#F59E0B",
          "#EF4444",
          "#8B5CF6",
          "#06B6D4",
          "#EC4899",
          "#10B981",
          "#F97316",
          "#6366F1",
          "#84CC16",
          "#F43F5E",
          "#06B6D4",
          "#10B981",
          "#F59E0B",
          "#EF4444",
        ],
        line: {
          color: "#333",
          width: 1,
        },
      },
      text: scores.map((score) => (score * 100).toFixed(1) + "%"),
      textposition: "auto",
      textfont: {
        size: 12,
        color: "#333",
        family: "monospace",
      },
      hovertemplate: "%{x} (%{y:.1%})<extra></extra>",
    },
  ];

  const layout = {
    xaxis: {
      title: "Emotions",
      titlefont: {
        size: 14,
        color: "#333",
        family: "monospace",
      },
      tickfont: {
        size: 12,
        color: "#333",
        family: "monospace",
      },
    },
    yaxis: {
      title: "Confidence Score",
      titlefont: {
        size: 14,
        color: "#333",
        family: "monospace",
      },
      tickfont: {
        size: 12,
        color: "#333",
        family: "monospace",
      },
      range: [0, 1],
      tickformat: ".0%",
    },
    margin: {
      l: 30,
      r: 20,
      t: 20,
      b: 20,
    },
    plot_bgcolor: "rgba(0,0,0,0)",
    paper_bgcolor: "rgba(0,0,0,0)",
    showlegend: false,
    height: 400,
    hovermode: "closest",
    dragmode: false,
  };

  const config = {
    displayModeBar: false,
    modeBarButtonsToRemove: [
      "pan2d",
      "lasso2d",
      "select2d",
      "zoom2d",
      "zoomIn2d",
      "zoomOut2d",
      "autoScale2d",
      "resetScale2d",
    ],
    scrollZoom: false,
    editable: false,
    staticPlot: false,
  };

  return (
    <div className="w-full">
      {title && <SectionTitle title={title} />}
      {expressionStats && Object.keys(expressionStats).length > 0 ? (
        <Plot
          data={data}
          layout={layout}
          config={config}
          className="w-full"
          useResizeHandler={true}
        />
      ) : (
        <div className="text-center text-gray-500 py-8">
          No expression data available
        </div>
      )}
    </div>
  );
};

export default ExpressionStats;
