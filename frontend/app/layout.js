import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], display: "swap" });

export const metadata = {
  title: "AgriSense Intelligence Hub",
  description: "AI-powered agricultural price forecasting, crop recommendation, and disease detection.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${inter.className} h-full`}>
      <body className="h-full overflow-hidden">{children}</body>
    </html>
  );
}
