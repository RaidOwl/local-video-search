import { NavLink } from 'react-router-dom'

export default function NavBar() {
  return (
    <nav>
      <div className="nav-inner">
        <NavLink to="/" className="nav-logo">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M15 10l4.553-2.277A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span className="nav-logo-text">Local Video Search</span>
        </NavLink>
        <div className="nav-links">
          <NavLink to="/" end className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>
            Search
          </NavLink>
          <NavLink to="/videos" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>
            Videos
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>
            Settings
          </NavLink>
          <a href="/docs" target="_blank" rel="noopener noreferrer" className="nav-link">
            API
          </a>
        </div>
      </div>
    </nav>
  )
}
