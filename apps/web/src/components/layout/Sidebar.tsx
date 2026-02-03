import Link from "next/link";

const items = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/signals", label: "Signals" },
  { href: "/positions", label: "Positions" },
  { href: "/orders", label: "Orders" },
  { href: "/training", label: "Training" },
  { href: "/risk", label: "Risk" },
  { href: "/data-quality", label: "Data Quality" },
  { href: "/settings", label: "Settings" },
];

export default function Sidebar() {
  return (
    <aside className="w-64 bg-ink text-mist px-6 py-8 hidden lg:flex flex-col">
      <div className="text-2xl font-display tracking-tight">TA Ops</div>
      <p className="text-sm text-mist/70 mt-2">Shadow-first trading control room</p>
      <nav className="mt-10 flex flex-col gap-3">
        {items.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className="text-sm uppercase tracking-wide text-mist/80 hover:text-ember transition"
          >
            {item.label}
          </Link>
        ))}
      </nav>
    </aside>
  );
}
