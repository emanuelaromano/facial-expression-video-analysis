import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import SectionTitle from "./sectionTitle";

const TranscriptAnalysis = ({ transcriptAnalysis, title }) => {
  return (
    <div className="w-full flex flex-col justify-center">
      <SectionTitle title={title} />
      <div className="prose prose-pink max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({ children, ...props }) => (
              <p className="mb-4" {...props}>
                {children}
              </p>
            ),
          }}
        >
          {transcriptAnalysis}
        </ReactMarkdown>
      </div>
    </div>
  );
};

export default TranscriptAnalysis;
