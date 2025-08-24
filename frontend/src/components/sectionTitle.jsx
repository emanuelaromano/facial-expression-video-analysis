const SectionTitle = ({ title }) => {
  return (
    <div className="flex items-center gap-3 my-4 border-t border-gray-200 pt-4">
      <h2 className="text-xl font-bold text-gray-800 tracking-wide">
        {title || "Title"}
      </h2>
    </div>
  );
};

export default SectionTitle;