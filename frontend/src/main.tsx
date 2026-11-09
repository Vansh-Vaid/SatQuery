import "@fontsource-variable/fraunces/opsz.css";
import "@fontsource-variable/fraunces/opsz-italic.css";
import "@fontsource/manrope/400.css";
import "@fontsource/manrope/500.css";
import "@fontsource/manrope/600.css";
import "@fontsource/manrope/700.css";
import "@fontsource/manrope/800.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import "./styles/tokens.css";
import "./styles/app.css";
import "./landing/landing.css";
import "./landing/motion.css";
import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";

// "/" is the landing page; "/console" is the analysis HUD. Each loads only its own code.
const Landing = lazy(() => import("./landing/Landing"));
const App = lazy(() => import("./App"));
const isConsole = window.location.pathname.startsWith("/console");
document.title = isConsole ? "SatQuery AI · Console" : "SatQuery AI · Ask the planet a question";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Suspense fallback={<div className="boot" aria-busy="true">SATQUERY AI</div>}>
      {isConsole ? <App /> : <Landing />}
    </Suspense>
  </StrictMode>,
);
