.PHONY: help build push dev dev-down logs

# Configuration
DOCKER ?= docker
COMPOSE ?= $(DOCKER) compose
IMAGE_REPO ?= registry.i.selfomy.com/one/paddleocr
IMAGE_TAG ?= latest
PORT ?= 8000

help:
	@echo "Available targets:"
	@echo "  make build              Build Docker image"
	@echo "  make push               Build and push Docker image to registry"
	@echo "  make dev                Run development environment with hot reload"
	@echo "  make dev-down           Stop development environment"
	@echo "  make logs               View development logs"

build:
	@echo "Building Docker image $(IMAGE_REPO):$(IMAGE_TAG)..."
	$(DOCKER) build --platform linux/amd64 -t $(IMAGE_REPO):$(IMAGE_TAG) -f Dockerfile .
	@echo "Build complete: $(IMAGE_REPO):$(IMAGE_TAG)"

push: build
	@echo "Pushing image $(IMAGE_REPO):$(IMAGE_TAG) to registry..."
	$(DOCKER) push $(IMAGE_REPO):$(IMAGE_TAG)
	@echo "Successfully pushed $(IMAGE_REPO):$(IMAGE_TAG)"

dev:
	@echo "Starting development environment..."
	$(COMPOSE) -f docker-compose.dev.yml --env-file .env up --build

dev-down:
	@echo "Stopping development environment..."
	$(COMPOSE) -f docker-compose.dev.yml down

logs:
	@echo "Tailing development logs..."
	$(COMPOSE) -f docker-compose.dev.yml logs -f --tail=200
