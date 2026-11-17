FROM node:22-slim

ENV NODE_ENV=production \
    PORT=3000

WORKDIR /app

COPY package.json ./
RUN npm install

COPY tsconfig.json server.ts ./
COPY src ./src
COPY frontend ./frontend

RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
