import Navbar from "./navbar";
import Footer from "./footer";
import { Outlet } from "react-router-dom";
import { useSelector } from "react-redux";
import Banner from "../components/banner";

const Layout = () => {
  const banner = useSelector((state) => state.video.banner);

  return (
    <div className="flex flex-col">
      <Navbar />
      {banner && <Banner message={banner.message} type={banner.type} />}
      <div className="flex-1 my-40">
        <Outlet />
      </div>
      <Footer />
    </div>
  );
};

export default Layout;
