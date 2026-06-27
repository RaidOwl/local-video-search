import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { useState, useEffect, useRef } from 'react'
import SearchPage from './pages/SearchPage'
import VideosPage from './pages/VideosPage'
import VideoDetailPage from './pages/VideoDetailPage'
import SettingsPage from './pages/SettingsPage'
import NavBar from './components/NavBar'
import { api } from './services/api'
import axios from 'axios'

function App() {
  const [health, setHealth] = useState<boolean | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    const check = () => {
      api.health()
        .then(() => setHealth(true))
        .catch(err => { if (!axios.isCancel(err)) setHealth(false) })
    }
    check()
    intervalRef.current = setInterval(check, 5000)
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [])

  return (
    <BrowserRouter>
      <div className="app">
        <NavBar />
        <main>
          {health === false && (
            <div className="warning">
              ⚠️ API is not available. Make sure the backend is running.
            </div>
          )}
          <Routes>
            <Route path="/" element={<SearchPage />} />
            <Route path="/videos" element={<VideosPage />} />
            <Route path="/videos/:id" element={<VideoDetailPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App