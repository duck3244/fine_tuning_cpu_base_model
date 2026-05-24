import { Link, NavLink, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Train from "./pages/Train";
import Evaluate from "./pages/Evaluate";
import Models from "./pages/Models";

const navItems = [
  { to: "/", label: "대시보드", end: true },
  { to: "/train", label: "학습" },
  { to: "/evaluate", label: "평가" },
  { to: "/models", label: "모델" },
];

export default function App() {
  return (
    <div className="flex h-full flex-col bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Link to="/" className="text-lg font-semibold">
            Fine-tuning CPU MVP
          </Link>
          <nav className="flex gap-4 text-sm">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `rounded px-3 py-1.5 transition ${
                    isActive
                      ? "bg-slate-900 text-white"
                      : "text-slate-600 hover:bg-slate-100"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/train" element={<Train />} />
          <Route path="/evaluate" element={<Evaluate />} />
          <Route path="/models" element={<Models />} />
        </Routes>
      </main>
    </div>
  );
}
