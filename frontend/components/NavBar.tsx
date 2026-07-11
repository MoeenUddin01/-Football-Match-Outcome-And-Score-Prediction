"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Predictor" },
  { href: "/rankings", label: "Rankings" },
  { href: "/scorers", label: "Scorers" },
  { href: "/validation", label: "Validation" },
];

export default function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-white/10 bg-surface/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <Link href="/" className="text-xl font-bold tracking-tight">
          <span className="text-primary">Chrono</span>Pitch
        </Link>
        <div className="flex gap-8">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`text-sm font-medium transition-colors ${
                pathname === link.href
                  ? "text-white"
                  : "text-muted hover:text-white"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>
        <div className="w-20" />
      </div>
    </nav>
  );
}
