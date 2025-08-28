import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <div className="fixed rounded-full text-[var(--primary-text)] top-0 left-0 right-0 z-50 mt-4 ml-4">
      <h1 className="text-2xl">
        <Link to="/">Great Speeches</Link>
      </h1>
    </div>
  );
};

export default Navbar;
