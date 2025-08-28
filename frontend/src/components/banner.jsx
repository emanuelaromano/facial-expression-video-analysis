const Banner = ({ message, type }) => {
  return (
    <div
      className={`${message ? "right-0" : "right-[-50%]"} ${type === "success" ? "text-green-500" : type === "info" ? "text-orange-400" : "text-red-500"} bg-white transition-all duration-300 ease-in-out mt-4 mr-4 fixed top-0 z-50 p-4 rounded-lg`}
    >
      <h1>
        <span className="font-bold">
          {type === "success" ? "Success" : type === "info" ? "Info" : "Error"}:
        </span>{" "}
        {message}
      </h1>
    </div>
  );
};

export default Banner;
