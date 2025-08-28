import VideoUpload from "./views/videoUpload";
import { Routes, Route, BrowserRouter } from "react-router-dom";
import Layout from "./layout/layout";
import LandingPage from "./views/landingPage";
import NotFound from "./views/notFound";
import Temp from "./views/temp";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route path="/" element={<LandingPage />} />
          <Route path="/video" element={<VideoUpload />} />
          <Route path="/temp" element={<Temp />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
