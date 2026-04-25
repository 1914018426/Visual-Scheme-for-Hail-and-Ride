# ========== Build Stage ==========
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json ./

# 使用国内 npm 镜像加速构建
RUN npm config set registry https://registry.npmmirror.com

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build the application
RUN npm run build

# ========== Production Stage ==========
FROM nginx:alpine

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx-default.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
