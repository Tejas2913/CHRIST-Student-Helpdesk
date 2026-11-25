/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        christ: {
          maroon: "#7A1B28",
          dark: "#54111B",
          deep: "#3D0C13",
          light: "#FDF2F4",
          surface: "#FAF5F5",
          gold: "#C5A059",
          goldLight: "#F5EFE6",
        }
      }
    },
  },
  plugins: [],
}
