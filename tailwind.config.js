/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/**/*.html"],
  theme: {
    extend: {
      colors: {
        page: "#F7F5F2",
        ink: "#2D2A26",
        sand: {
          DEFAULT: "#E2DED6",
          light: "#F0EDE8",
        },
        yt: {
          DEFAULT: "#FF0000",
          dark: "#CC0000",
          light: "#FEE2E2",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
      },
      typography: ({ theme }) => ({
        DEFAULT: {
          css: {
            "--tw-prose-body": theme("colors.stone.700"),
            "--tw-prose-headings": theme("colors.stone.900"),
            "--tw-prose-lead": theme("colors.stone.600"),
            "--tw-prose-links": "#CC0000",
            "--tw-prose-bold": theme("colors.stone.900"),
            "--tw-prose-quotes": theme("colors.stone.700"),
            "--tw-prose-quote-borders": theme("colors.stone.300"),
            "--tw-prose-counters": theme("colors.stone.500"),
            "--tw-prose-bullets": theme("colors.stone.400"),
            "--tw-prose-hr": theme("colors.stone.200"),
            "--tw-prose-th-borders": theme("colors.stone.300"),
            "--tw-prose-td-borders": theme("colors.stone.200"),
            "--tw-prose-code": theme("colors.stone.800"),
            maxWidth: "none",
          },
        },
      }),
    },
  },
  plugins: [
    require("@tailwindcss/typography"),
    require("@tailwindcss/forms"),
  ],
};
