import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { useState, useEffect } from 'react'
import SearchPage from './pages/SearchPage'
import VideosPage from './pages/VideosPage'
import VideoDetailPage from './pages/VideoDetailPage'
import NavBar from './components/NavBar'
import { api } from './services/api'

function App() {
  const [health, setHealth] = useState<boolean>(false)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(false))
  }, [])

  return (
    <BrowserRouter>
      <div className="app">
        <NavBar />
        <main>
          {!health && (
            <div className="warning">
              ⚠️ API is not available. Make sure the backend is running.
            </div>
          )}
          <Routes>
            <Route path="/" element={<SearchPage />} />
            <Route path="/videos" element={<VideosPage />} />
            <Route path="/videos/:id" element={<VideoDetailPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App