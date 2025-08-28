import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import store from "./redux/store.jsx";
import { Provider } from "react-redux";

// Disable Tab key navigation globally
document.addEventListener("keydown", (event) => {
  if (event.key === "Tab") {
    event.preventDefault();
    event.stopPropagation();
    return false;
  }
});

// Also prevent Tab key on keyup and keypress
document.addEventListener("keyup", (event) => {
  if (event.key === "Tab") {
    event.preventDefault();
    event.stopPropagation();
    return false;
  }
});

document.addEventListener("keypress", (event) => {
  if (event.key === "Tab") {
    event.preventDefault();
    event.stopPropagation();
    return false;
  }
});

createRoot(document.getElementById("root")).render(
  <Provider store={store}>
    <App />
  </Provider>,
);
