import OrdersTable from "@/components/tables/OrdersTable";

export default function OrdersPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-display">Orders</h1>
      <OrdersTable />
    </div>
  );
}
