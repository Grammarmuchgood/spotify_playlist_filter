import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// One QueryClient for the whole app - it's TanStack Query's actual cache
// and fetch-coordinator, created once here rather than inside App so it
// survives App re-rendering (creating a new one on every render would
// throw away all cached data and in-flight request tracking each time).
const queryClient = new QueryClient()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)
