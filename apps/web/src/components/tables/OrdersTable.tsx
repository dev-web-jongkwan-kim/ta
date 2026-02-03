import { fetchJSON } from "@/lib/api";

interface Order {
  order_id: string;
  symbol: string;
  side: string;
  type: string;
  status: string;
  qty: number;
  price: number | null;
  created_at: string;
}

export default async function OrdersTable() {
  let orders: Order[] = [];
  try {
    orders = await fetchJSON("/api/orders");
  } catch (e) {
    console.error("Failed to fetch orders:", e);
  }

  const getStatusColor = (status: string) => {
    switch (status?.toUpperCase()) {
      case "FILLED":
        return "bg-emerald-100 text-emerald-700";
      case "NEW":
      case "OPEN":
        return "bg-blue-100 text-blue-700";
      case "CANCELED":
        return "bg-gray-100 text-gray-600";
      case "REJECTED":
        return "bg-red-100 text-red-700";
      default:
        return "bg-gray-100 text-gray-600";
    }
  };

  const getSideColor = (side: string) => {
    return side?.toUpperCase() === "BUY" ? "text-emerald-600" : "text-red-600";
  };

  return (
    <div className="rounded-2xl bg-white/80 p-6 shadow-sm border border-ink/5">
      <h3 className="font-display text-lg">Orders</h3>
      <table className="mt-4 w-full text-sm">
        <thead className="text-slate">
          <tr className="text-left">
            <th className="py-2">Order ID</th>
            <th className="py-2">Symbol</th>
            <th className="py-2">Side</th>
            <th className="py-2">Type</th>
            <th className="py-2">Qty</th>
            <th className="py-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {orders.length === 0 ? (
            <tr>
              <td colSpan={6} className="py-4 text-center text-slate">
                No orders found
              </td>
            </tr>
          ) : (
            orders.slice(0, 20).map((order) => (
              <tr key={order.order_id} className="border-t border-ink/5">
                <td className="py-3 font-display text-xs">{order.order_id}</td>
                <td className="py-3">{order.symbol}</td>
                <td className={`py-3 font-medium ${getSideColor(order.side)}`}>
                  {order.side}
                </td>
                <td className="py-3 text-slate">{order.type}</td>
                <td className="py-3 text-slate">{order.qty}</td>
                <td className="py-3">
                  <span className={`px-2 py-1 rounded-full text-xs ${getStatusColor(order.status)}`}>
                    {order.status}
                  </span>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
