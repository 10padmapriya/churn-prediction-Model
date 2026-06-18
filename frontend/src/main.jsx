import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import ChurnDashboard from "./ChurnDashboard";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ChurnDashboard />
  </StrictMode>
);
