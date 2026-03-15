const App = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-4">
      <h1 className="text-4xl font-bold text-primary-500">
        TaxDiatoms
      </h1>
      <p className="text-slate-600">
        Ambiente configurado com sucesso: Vite + React 19 + Tailwind v4
      </p>
      <div className="flex gap-2">
        <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
          TypeScript Strict
        </span>
        <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
          React 19
        </span>
      </div>
    </div>
  )
}

export default App
