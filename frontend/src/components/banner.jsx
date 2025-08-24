const Banner = ({ message, type }) => {
  return (
    <div
      className={`${message ? "right-0" : "right-[-50%]"} ${type === "success" ? "bg-green-500" : type === "info" ? "bg-orange-400" : "bg-red-500"} transition-all duration-300 ease-in-out mt-4 mr-4 fixed top-0 z-50 text-white p-4 rounded-lg`}
    >
      <h1>{message}</h1>
    </div>
  );
};

export default Banner;
