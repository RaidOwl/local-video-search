import { NavLink } from 'react-router-dom'

export default function NavBar() {
  return (
    <nav>
      <div className="nav-inner">
        <NavLink to="/" className="nav-logo">
          <span>🎬</span> Local Video Search
        </NavLink>
        <div className="nav-links">
          <NavLink to="/" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>
            Search
          </NavLink>
          <NavLink to="/videos" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>
            Videos
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>
            Settings
          </NavLink>
        </div>
      </div>
    </nav>
  )
}
