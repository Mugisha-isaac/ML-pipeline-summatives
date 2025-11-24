# Audio Talent Classification Frontend

Modern Next.js frontend for ML-powered audio talent classification system.

## Technology Stack

- Next.js 14 with TypeScript
- Tailwind CSS for styling
- shadcn/ui component library
- Yarn for package management
- Axios for API integration
- React Hook Form for form handling

## Quick Start

### Installation

```bash
cd frontend
yarn install
```

### Development

```bash
yarn dev
```

Open http://localhost:3000 in your browser.

### Build

```bash
yarn build
yarn start
```

## Features

### Pages

- **Home** (`/`) - Main landing page with navigation
- **Predict** (`/predict`) - Single audio file prediction
- **Train** (`/train`) - Model retraining with data upload
- **Visualizations** (`/visualizations`) - Feature distribution charts

### Integration

Frontend communicates with backend at:
- Default: `http://localhost:8000`
- Configured via: `NEXT_PUBLIC_API_URL` environment variable

### Environment Variables

Create `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production:

```
NEXT_PUBLIC_API_URL=https://your-api-domain.com
```

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx           Root layout
│   ├── page.tsx             Home page
│   ├── globals.css          Global styles
│   ├── predict/
│   │   └── page.tsx         Prediction page
│   ├── train/
│   │   └── page.tsx         Training page
│   └── visualizations/
│       └── page.tsx         Visualizations page
├── lib/
│   ├── api.ts               API client functions
│   └── utils.ts             Utility functions
├── package.json             Dependencies
├── tsconfig.json            TypeScript config
├── tailwind.config.ts       Tailwind configuration
└── postcss.config.js        PostCSS configuration
```

## API Integration

The frontend connects to backend endpoints:

- `GET /health` - Service health check
- `GET /model-info` - Model information
- `POST /predictions/single` - Single audio prediction
- `POST /predictions/batch` - Batch predictions
- `POST /upload-data` - Upload training data
- `POST /retrain` - Start model retraining
- `GET /train-status` - Check training progress
- `GET /model-metrics` - Get model performance metrics
- `GET /visualizations/mfcc` - MFCC feature distribution
- `GET /visualizations/spectral` - Spectral features
- `GET /visualizations/feature-info` - Feature interpretations

## Styling

Uses Tailwind CSS with custom configuration:

- Dark theme with slate colors
- Responsive grid layouts
- Form inputs with Tailwind styling
- Smooth transitions and hover effects

### Custom CSS Classes

Global styles in `app/globals.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

## Type Safety

Full TypeScript support with:

- Type definitions for API responses
- Strict mode enabled
- Path aliases configured (`@/*`)

## Development Tips

### Adding New Pages

1. Create directory: `app/new-page/`
2. Create file: `app/new-page/page.tsx`
3. Add to navigation in `app/page.tsx`

### Adding Components

1. Create file: `components/ComponentName.tsx`
2. Use Client Components with `'use client'` directive for interactivity
3. Import and use in pages

### Calling Backend APIs

```typescript
const response = await fetch(
  `${process.env.NEXT_PUBLIC_API_URL}/api/v1/endpoint`,
  {
    method: 'POST',
    body: JSON.stringify(data),
  }
)
```

## Deployment

### Vercel (Recommended)

```bash
vercel
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile
COPY . .
RUN yarn build
CMD ["yarn", "start"]
```

Build and run:

```bash
docker build -t talent-frontend .
docker run -p 3000:3000 talent-frontend
```

### Environment Variables for Production

Set on deployment platform:

```
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

## Troubleshooting

### Port 3000 already in use
```bash
yarn dev -p 3001
```

### Module not found errors
```bash
rm -rf node_modules yarn.lock
yarn install
```

### API connection errors
- Check backend is running on configured URL
- Verify `NEXT_PUBLIC_API_URL` environment variable
- Check CORS settings in backend

## Performance Optimization

- Image optimization with Next.js Image component
- Code splitting and lazy loading
- CSS minification with Tailwind
- Automatic static optimization

## Version

Version: 1.0.0
Last Updated: November 24, 2025
Status: Production Ready
