import VideoUpload from './views/videoUpload'
import { Routes, Route, BrowserRouter } from 'react-router-dom'
import Layout from './layout/layout'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route path="/" element={<VideoUpload />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
