(function () {
  "use strict";

  const CATEGORY = {
    player: 0x0001,
    platform: 0x0002,
    hazard: 0x0004,
    collectible: 0x0008,
  };
  const HOLOGRAM_COLORS = {
    platform: 0x00ffcc,
    hazard: 0xff3333,
    collectible: 0xffd700,
  };

  function choosePlayerStart(sceneData) {
    const width = sceneData.canvas.width;
    const height = sceneData.canvas.height;
    const provided = sceneData.rules.player_start || {};
    const start = {
      x: Phaser.Math.Clamp(Number(provided.x) || 0, 0, width),
      y: Phaser.Math.Clamp(Number(provided.y) || 0, 0, height),
    };
    const platforms = sceneData.rules.platforms
      .map((entry) => {
        const object = sceneData.objects[entry.object_index];
        const [left, top, right, bottom] = object.bbox || [];
        const vertices = object.physics;
        const finiteVertices = Array.isArray(vertices) && vertices.every((point) => (
          point && Number.isFinite(point.x) && Number.isFinite(point.y)
        ));
        const xs = finiteVertices ? vertices.map((point) => point.x) : [left, right];
        const ys = finiteVertices ? vertices.map((point) => point.y) : [top, bottom];
        if (![...xs, ...ys].every(Number.isFinite)) return null;
        const minY = Math.min(...ys);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        return {
          centerX: (minX + maxX) / 2,
          topY: minY,
          centerY: (minY + Math.max(...ys)) / 2,
          width: maxX - minX,
          area: Number(object.area_ratio) || Math.max(0, right - left) * Math.max(0, bottom - top),
        };
      })
      .filter(Boolean);
    if (!platforms.length) return start;

    if (sceneData.meta?.rules_fallback) {
      const widest = platforms.reduce((best, platform) => (
        platform.width > best.width ? platform : best
      ));
      return {
        x: Phaser.Math.Clamp(widest.centerX, 0, width),
        y: Phaser.Math.Clamp(widest.topY - 50, 0, height),
      };
    }

    const nearest = platforms.reduce((best, platform) => (
      Math.hypot(platform.centerX - start.x, platform.centerY - start.y)
        < Math.hypot(best.centerX - start.x, best.centerY - start.y)
        ? platform
        : best
    ));
    if (start.y < nearest.topY - 100) {
      const largest = platforms.reduce((best, platform) => (
        platform.area > best.area ? platform : best
      ));
      return {
        x: Phaser.Math.Clamp(largest.centerX, 0, width),
        y: Phaser.Math.Clamp(largest.topY - 50, 0, height),
      };
    }
    return start;
  }

  function assertSceneData(sceneData) {
    if (!sceneData || !Array.isArray(sceneData.objects) || !sceneData.rules) {
      throw new TypeError("Scene data must include objects and rules.");
    }
    const { width, height } = sceneData.canvas || {};
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      throw new TypeError("Scene data must include positive canvas dimensions.");
    }
    for (const role of ["platforms", "hazards", "collectibles"]) {
      if (!Array.isArray(sceneData.rules[role])) {
        throw new TypeError(`Scene rules are missing the ${role} list.`);
      }
      for (const item of sceneData.rules[role]) {
        if (!Number.isInteger(item.object_index) || !sceneData.objects[item.object_index]) {
          throw new TypeError(`${role} contains an invalid object_index.`);
        }
      }
    }
  }

  function toMatterVertices(object) {
    let vertices = object.physics;
    const hasValidCoordinates = Array.isArray(vertices)
      && vertices.length >= 3
      && vertices.every((point) => (
        point
        && typeof point.x === "number"
        && Number.isFinite(point.x)
        && typeof point.y === "number"
        && Number.isFinite(point.y)
      ));

    if (!hasValidCoordinates) {
      const error = new TypeError("Physics polygon coordinates must be finite numbers.");
      console.error(
        `Invalid physics polygon for ${object.label || "object"}; falling back to its bbox rectangle.`,
        error,
      );
      if (!Array.isArray(object.bbox) || object.bbox.length !== 4) {
        throw new TypeError("Cannot build a Matter rectangle without a four-value bbox.");
      }
      const [leftValue, topValue, rightValue, bottomValue] = object.bbox;
      const left = Number(leftValue);
      const top = Number(topValue);
      const right = Number(rightValue);
      const bottom = Number(bottomValue);
      if (![left, top, right, bottom].every(Number.isFinite) || right <= left || bottom <= top) {
        throw new TypeError("Cannot build a Matter rectangle from an invalid bbox.");
      }
      vertices = [
        { x: left, y: top },
        { x: right, y: top },
        { x: right, y: bottom },
        { x: left, y: bottom },
      ];
    }

    const center = vertices.reduce(
      (sum, point) => ({ x: sum.x + point.x / vertices.length, y: sum.y + point.y / vertices.length }),
      { x: 0, y: 0 },
    );
    return {
      center,
      local: vertices.map((point) => ({
        x: point.x - center.x,
        y: point.y - center.y,
      })),
    };
  }

  function removeFlatBackground(dataUrl) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        const context = canvas.getContext("2d", { willReadFrequently: true });
        if (!context) {
          reject(new Error("Could not prepare restyled sprite transparency."));
          return;
        }
        context.drawImage(image, 0, 0);
        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        const { data } = imageData;
        const width = canvas.width;
        const height = canvas.height;
        const total = width * height;
        const corners = [0, width - 1, (height - 1) * width, total - 1];
        const backgrounds = new Set();
        for (const index of corners) {
          const offset = index * 4;
          if (data[offset + 3] < 16) continue;
          if (Math.max(data[offset], data[offset + 1], data[offset + 2]) <= 48) {
            backgrounds.add(0);
          } else if (Math.min(data[offset], data[offset + 1], data[offset + 2]) >= 207) {
            backgrounds.add(255);
          }
        }
        for (const background of backgrounds) {
          const visited = new Uint8Array(total);
          const stack = [];
          for (const index of corners) {
            const offset = index * 4;
            const distance = Math.max(
              Math.abs(data[offset] - background),
              Math.abs(data[offset + 1] - background),
              Math.abs(data[offset + 2] - background),
            );
            if (data[offset + 3] > 0 && distance <= 72 && !visited[index]) {
              visited[index] = 1;
              stack.push(index);
            }
          }
          while (stack.length) {
            const index = stack.pop();
            const offset = index * 4;
            const distance = Math.max(
              Math.abs(data[offset] - background),
              Math.abs(data[offset + 1] - background),
              Math.abs(data[offset + 2] - background),
            );
            if (distance > 72) continue;
            data[offset + 3] = Math.min(
              data[offset + 3],
              Math.round(255 * Math.max(0, (distance - 48) / 24)),
            );
            const x = index % width;
            const y = Math.floor(index / width);
            for (const next of [
              x > 0 ? index - 1 : -1,
              x + 1 < width ? index + 1 : -1,
              y > 0 ? index - width : -1,
              y + 1 < height ? index + width : -1,
            ]) {
              if (next < 0 || visited[next]) continue;
              const nextOffset = next * 4;
              const nextDistance = Math.max(
                Math.abs(data[nextOffset] - background),
                Math.abs(data[nextOffset + 1] - background),
                Math.abs(data[nextOffset + 2] - background),
              );
              if (data[nextOffset + 3] > 0 && nextDistance <= 72) {
                visited[next] = 1;
                stack.push(next);
              }
            }
          }
        }
        context.putImageData(imageData, 0, 0);
        resolve(canvas.toDataURL("image/png"));
      };
      image.onerror = () => reject(new Error("Could not decode restyled sprite."));
      image.src = dataUrl;
    });
  }

  function createGame(sceneData, assets) {
    if (typeof window.Phaser === "undefined") {
      throw new Error("Phaser could not be loaded. Check the connection and reload the page.");
    }
    assertSceneData(sceneData);

    const gameRoot = document.getElementById("game");
    const placeholder = document.getElementById("game-placeholder");
    const scoreElement = document.getElementById("score");
    const winElement = document.getElementById("win-condition");
    const { width, height } = sceneData.canvas;
    if (placeholder) placeholder.remove();

    let score = 0;
    const collected = new Set();
    const collectiblesById = new Map();
    const savedPlayerStart = choosePlayerStart(sceneData);

    class RoomScene extends Phaser.Scene {
      constructor() {
        super({ key: "Room2GameScene" });
      }

      preload() {
        if (sceneData.canvas_base64) {
          this.load.image("scene-photo-background", sceneData.canvas_base64);
        }
        this.load.spritesheet(
          "room2game-character",
          "https://labs.phaser.io/assets/sprites/dude.png",
          { frameWidth: 32, frameHeight: 48 },
        );
      }

      create() {
        this.playerStart = savedPlayerStart;
        this.matter.world.setBounds(0, 0, width, height, 64, true, true, false, true);
        this.matter.world.setGravity(0, 1.05);
        this.cameras.main.setBackgroundColor("#171b14");
        this.lights.enable().setAmbientColor(0x888888);
        this.playerLight = this.lights.addLight(
          this.playerStart?.x || width * 0.1,
          this.playerStart?.y || height * 0.5,
          Math.max(width, height) * 0.45,
          0xffe6a3,
          1.1,
        );
        if (this.textures.exists("scene-photo-background")) {
          this.add.image(0, 0, "scene-photo-background")
            .setOrigin(0, 0)
            .setDepth(-1)
            .setDisplaySize(width, height);
        }

        this.levelGraphics = this.add.graphics().setDepth(2);
        this.shadowGraphics = this.add.graphics().setDepth(1);
        this.playerGraphics = this.add.graphics().setDepth(5);
        this.playerAuraGraphics = this.add.graphics().setDepth(4);
        this.spriteImages = new Map();
        this.objectBodies = new Map();
        this.drawObjects(sceneData.objects, "platform");
        this.drawObjects(sceneData.objects, "hazard");
        this.drawObjects(sceneData.objects, "collectible");
        this.drawOptionalAssets(assets || {});

        this.platformBodies = new Set();
        this.hazardBodies = new Set();
        this.collectibleBodies = new Set();
        this.bodiesById = new Map();
        this.createRoleBodies(sceneData.rules.platforms, "platform");
        this.createRoleBodies(sceneData.rules.hazards, "hazard");
        this.createRoleBodies(sceneData.rules.collectibles, "collectible");

        this.playerBody = this.matter.add.rectangle(
          this.playerStart.x,
          this.playerStart.y,
          Math.max(18, width * 0.025),
          Math.max(24, height * 0.06),
          {
            label: "room2game-player",
            friction: 0,
            frictionAir: 0.01,
            restitution: 0,
            chamfer: { radius: 5 },
            collisionFilter: {
              category: CATEGORY.player,
              mask: CATEGORY.platform | CATEGORY.hazard | CATEGORY.collectible,
            },
          },
        );
        this.playerBody.label = "room2game-player";
        this.playerStart.x = this.playerBody.position.x;
        this.playerStart.y = this.playerBody.position.y;
        this.playerSprite = this.textures.exists("room2game-character")
          ? this.add.sprite(
            this.playerBody.position.x,
            this.playerBody.position.y,
            "room2game-character",
            4,
          ).setDepth(5).setPipeline("Light2D")
          : null;
        if (this.playerSprite) {
          this.anims.create({
            key: "room2game-walk-left",
            frames: this.anims.generateFrameNumbers("room2game-character", {
              start: 0,
              end: 3,
            }),
            frameRate: 10,
            repeat: -1,
          });
          this.anims.create({
            key: "room2game-walk-right",
            frames: this.anims.generateFrameNumbers("room2game-character", {
              start: 5,
              end: 8,
            }),
            frameRate: 10,
            repeat: -1,
          });
          this.anims.create({
            key: "room2game-idle",
            frames: [{ key: "room2game-character", frame: 4 }],
            frameRate: 1,
            repeat: -1,
          });
        }

        this.keys = this.input.keyboard.addKeys("W,A,S,D,SPACE,UP,LEFT,RIGHT");
        this.canJump = false;
        this.resetAfterHazard = false;

        this.matter.world.on("collisionstart", this.handleCollisions, this);
        this.matter.world.on("collisionactive", this.handleActiveCollisions, this);
        this.events.once(Phaser.Scenes.Events.SHUTDOWN, this.onShutdown, this);

        this.input.on("pointerdown", () => {
          this.jump();
        });

        if (winElement) {
          winElement.textContent = sceneData.rules.win_condition || "Reach the end.";
        }
        if (scoreElement) scoreElement.textContent = "0 points";
        this.requestRestyledSprites();
      }

      drawObjects(objects, role) {
        const roleEntries = sceneData.rules[`${role}s`];
        roleEntries.forEach((entry) => {
          const object = objects[entry.object_index];
          const points = object.display;
          if (!Array.isArray(points) || points.length < 3) return;
          const color = HOLOGRAM_COLORS[role];
          this.levelGraphics.fillStyle(color, 0.12);
          this.levelGraphics.fillPoints(points, true);
          this.levelGraphics.lineStyle(12, color, 0.14);
          this.levelGraphics.strokePoints(points, true);
          this.levelGraphics.lineStyle(4, color, 0.9);
          this.levelGraphics.strokePoints(points, true);
        });
      }

      drawOptionalAssets(assetMap) {
        for (const object of sceneData.objects) {
          const key = assetMap[object.object_index];
          if (!key || !this.textures.exists(key) || !Array.isArray(object.bbox)) continue;
          const [left, top, right, bottom] = object.bbox;
          const sprite = this.add.image((left + right) / 2, (top + bottom) / 2, key)
            .setOrigin(0.5)
            .setDisplaySize(Math.max(1, right - left), Math.max(1, bottom - top))
            .setDepth(1.5)
            .setPipeline("Light2D");
          this.trackObjectSprite(object.object_index, sprite);
        }
      }

      trackObjectSprite(objectIndex, sprite) {
        const body = this.objectBodies.get(objectIndex);
        if (body) {
          this.spriteOffsets = this.spriteOffsets || new Map();
          this.spriteOffsets.set(objectIndex, {
            x: sprite.x - body.position.x,
            y: sprite.y - body.position.y,
          });
        }
        this.spriteImages.set(objectIndex, sprite);
      }

      requestRestyledSprites() {
        if (!sceneData.image_id) return;
        const objectIndices = new Set();
        for (const role of ["platforms", "hazards", "collectibles"]) {
          for (const entry of sceneData.rules[role]) {
            const object = sceneData.objects[entry.object_index];
            if (object.label !== "ground") objectIndices.add(entry.object_index);
          }
        }
        for (const objectIndex of objectIndices) {
          this.loadRestyledSprite(objectIndex);
        }
      }

      async loadRestyledSprite(objectIndex) {
        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), 5000);
        try {
          const apiOrigin = window.location.port === "8000"
            ? ""
            : "http://localhost:8000";
          const response = await fetch(`${apiOrigin}/restyle`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              image_id: sceneData.image_id,
              object_index: objectIndex,
              theme: sceneData.rules.theme || "pixel art",
            }),
            signal: controller.signal,
          });
          if (!response.ok) {
            const payload = await response.json().catch(() => ({}));
            throw new Error(payload.detail || `Restyle request failed (${response.status}).`);
          }
          const payload = await response.json();
          if (typeof payload.png_base64 !== "string" || !payload.png_base64) {
            throw new Error("Restyle response did not contain a PNG.");
          }
          if (!this.sys.isActive()) return;

          const themeKey = (sceneData.rules.theme || "pixel-art")
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/^-|-$/g, "");
          const textureKey = `restyle-${sceneData.image_id}-${objectIndex}-${themeKey}`;
          const addSprite = () => {
            if (!this.sys.isActive() || this.spriteImages.has(objectIndex)) return;
            const object = sceneData.objects[objectIndex];
            const [left, top, right, bottom] = object.bbox;
            const sprite = this.add.image(
              (left + right) / 2,
              (top + bottom) / 2,
              textureKey,
            ).setOrigin(0.5);
            sprite.setDisplaySize(
              Math.max(1, right - left),
              Math.max(1, bottom - top),
            ).setDepth(1.5).setPipeline("Light2D");
            this.trackObjectSprite(objectIndex, sprite);
            this.redrawLevel();
          };

          if (this.textures.exists(textureKey)) {
            addSprite();
          } else {
            const clearTextureListeners = () => {
              this.textures.off(Phaser.Textures.Events.LOAD, onTextureLoad);
              this.textures.off(Phaser.Textures.Events.ERROR, onTextureError);
              window.clearTimeout(textureTimeout);
            };
            const onTextureLoad = (loadedKey) => {
              if (loadedKey !== textureKey) return;
              clearTextureListeners();
              addSprite();
            };
            const onTextureError = (failedKey) => {
              if (failedKey !== textureKey) return;
              clearTextureListeners();
              console.warn(
                `Restyle image could not be decoded for object ${objectIndex}; keeping its polygon fallback.`,
              );
            };
            const textureTimeout = window.setTimeout(() => {
              clearTextureListeners();
              console.warn(
                `Restyle image timed out for object ${objectIndex}; keeping its polygon fallback.`,
              );
            }, 5000);
            this.textures.on(Phaser.Textures.Events.LOAD, onTextureLoad);
            this.textures.on(Phaser.Textures.Events.ERROR, onTextureError);
            removeFlatBackground(`data:image/png;base64,${payload.png_base64}`)
              .then((transparentDataUrl) => {
                if (this.sys.isActive()) {
                  this.textures.addBase64(textureKey, transparentDataUrl);
                }
              })
              .catch((error) => {
                clearTextureListeners();
                console.warn(
                  `Restyle transparency processing failed for object ${objectIndex}; keeping its polygon fallback.`,
                  error,
                );
              });
          }
        } catch (error) {
          console.warn(
            `Restyle unavailable for object ${objectIndex}; keeping its polygon fallback.`,
            error,
          );
        } finally {
          window.clearTimeout(timeout);
        }
      }

      createRoleBodies(entries, role) {
        entries.forEach((entry) => {
          const object = sceneData.objects[entry.object_index];
          const { center, local } = toMatterVertices(object);
          const category = CATEGORY[role];
          const body = this.matter.add.fromVertices(
            center.x,
            center.y,
            [local],
            {
              label: `room2game-${role}-${entry.object_index}`,
              isStatic: true,
              isSensor: role !== "platform",
              friction: role === "platform" ? 0.85 : 0,
              collisionFilter: {
                category,
                mask: CATEGORY.player,
              },
            },
            true,
          );
          body.label = `room2game-${role}-${entry.object_index}`;
          this.objectBodies.set(entry.object_index, body);
          this.bodiesById.set(body.id, body);
          if (role === "platform") this.platformBodies.add(body.id);
          if (role === "hazard") this.hazardBodies.add(body.id);
          if (role === "collectible") {
            this.collectibleBodies.add(body.id);
            collectiblesById.set(body.id, entry);
          }
        });
      }

      handleCollisions(event) {
        for (const pair of event.pairs) {
          this.handlePair(pair.bodyA, pair.bodyB);
        }
      }

      handleActiveCollisions(event) {
        for (const pair of event.pairs) {
          this.handlePair(pair.bodyA, pair.bodyB);
        }
      }

      handlePair(bodyA, bodyB) {
        const player = bodyA.id === this.playerBody.id ? bodyA : bodyB.id === this.playerBody.id ? bodyB : null;
        if (!player) return;
        const other = player === bodyA ? bodyB : bodyA;
        if (this.platformBodies.has(other.id)) {
          this.canJump = true;
        } else if (this.hazardBodies.has(other.id)) {
          this.resetAfterHazard = true;
        } else if (this.collectibleBodies.has(other.id)) {
          this.collectCollectible(other.id);
        }
      }

      collectCollectible(bodyId) {
        if (collected.has(bodyId)) return;
        collected.add(bodyId);
        const entry = collectiblesById.get(bodyId);
        score += entry && Number.isFinite(entry.points) ? entry.points : 10;
        this.collectibleBodies.delete(bodyId);
        this.matter.world.remove(this.bodiesById.get(bodyId));
        if (entry && this.spriteImages.has(entry.object_index)) {
          this.spriteImages.get(entry.object_index).setVisible(false);
        }
        if (scoreElement) scoreElement.textContent = `${score} point${score === 1 ? "" : "s"}`;
        this.drawCollectedState();
      }

      redrawLevel() {
        this.levelGraphics.clear();
        this.drawObjects(sceneData.objects, "platform");
        this.drawObjects(sceneData.objects, "hazard");
        const collectedIndices = new Set(
          [...collected].map((id) => collectiblesById.get(id).object_index),
        );
        const entries = sceneData.rules.collectibles;
        const keepEntries = entries.filter((entry) => !collectedIndices.has(entry.object_index));
        this.drawObjectsForList(keepEntries, HOLOGRAM_COLORS.collectible);
      }

      drawCollectedState() {
        this.redrawLevel();
      }

      drawObjectsForList(entries, color) {
        for (const entry of entries) {
          const object = sceneData.objects[entry.object_index];
          if (!object || !Array.isArray(object.display)) continue;
          this.levelGraphics.fillStyle(color, 0.12);
          this.levelGraphics.fillPoints(object.display, true);
          this.levelGraphics.lineStyle(12, color, 0.14);
          this.levelGraphics.strokePoints(object.display, true);
          this.levelGraphics.lineStyle(4, color, 0.9);
          this.levelGraphics.strokePoints(object.display, true);
        }
      }

      jump() {
        if (!this.canJump) return;
        this.matter.body.setVelocity(this.playerBody, {
          x: this.playerBody.velocity.x,
          y: -Math.max(7, height * 0.018),
        });
        this.canJump = false;
      }

      resetPlayer() {
        this.matter.body.setPosition(this.playerBody, this.playerStart);
        this.matter.body.setVelocity(this.playerBody, { x: 0, y: 0 });
        this.matter.body.setAngularVelocity(this.playerBody, 0);
        if (this.playerSprite) {
          this.playerSprite.setPosition(this.playerStart.x, this.playerStart.y);
          this.playerSprite.play("room2game-idle", true);
        }
        this.resetAfterHazard = false;
        this.canJump = false;
      }

      resetGame() {
        for (const bodyId of collected) {
          const body = this.bodiesById.get(bodyId);
          if (body) this.matter.world.add(body);
          this.collectibleBodies.add(bodyId);
          const entry = collectiblesById.get(bodyId);
          if (entry && this.spriteImages.has(entry.object_index)) {
            this.spriteImages.get(entry.object_index).setVisible(true);
          }
        }
        collected.clear();
        score = 0;
        if (scoreElement) scoreElement.textContent = "0 points";
        this.drawCollectedState();
        this.resetPlayer();
      }

      onShutdown() {
        this.matter.world.off("collisionstart", this.handleCollisions, this);
        this.matter.world.off("collisionactive", this.handleActiveCollisions, this);
      }

      drawShadows() {
        this.shadowGraphics.clear();
        const player = this.playerBody;
        const platformBelow = [...this.platformBodies]
          .map((bodyId) => this.bodiesById.get(bodyId))
          .filter((body) => body && body.position.y >= player.position.y)
          .map((body) => ({
            body,
            distance: body.position.y - player.position.y,
            top: Math.min(...body.vertices.map((vertex) => vertex.y)),
          }))
          .sort((a, b) => a.distance - b.distance)[0];
        const fallDistance = platformBelow ? platformBelow.distance : height * 0.25;
        const scale = Phaser.Math.Clamp(1 - fallDistance / (height * 0.7), 0.2, 1);
        const shadowY = platformBelow
          ? platformBelow.top + 4
          : Math.min(height - 4, player.position.y + fallDistance);
        this.shadowGraphics.fillStyle(0x050805, 0.34 * scale);
        this.shadowGraphics.fillEllipse(
          player.position.x,
          shadowY,
          Math.max(16, width * 0.055 * scale),
          Math.max(5, height * 0.025 * scale),
        );
      }

      update() {
        const moveLeft = this.keys.LEFT.isDown || this.keys.A.isDown;
        const moveRight = this.keys.RIGHT.isDown || this.keys.D.isDown;
        const direction = Number(moveRight) - Number(moveLeft);
        const maxSpeed = Math.max(3, width * 0.009);
        this.matter.body.setVelocity(this.playerBody, {
          x: direction === 0 && Math.abs(this.playerBody.velocity.x) <= 0.2
            ? 0
            : direction * maxSpeed || this.playerBody.velocity.x * 0.78,
          y: this.playerBody.velocity.y,
        });

        if (
          Phaser.Input.Keyboard.JustDown(this.keys.SPACE)
          || Phaser.Input.Keyboard.JustDown(this.keys.UP)
          || Phaser.Input.Keyboard.JustDown(this.keys.W)
        ) {
          this.jump();
        }
        if (this.resetAfterHazard) this.resetPlayer();
        if (this.playerBody.position.y > height + 80) this.resetPlayer();

        const player = this.playerBody;
        this.playerGraphics.clear();
        this.playerAuraGraphics.clear();
        this.playerAuraGraphics.fillStyle(0x00ffcc, 0.15);
        this.playerAuraGraphics.fillCircle(
          player.position.x,
          player.position.y,
          Math.max(22, width * 0.035),
        );
        this.playerAuraGraphics.lineStyle(2, 0x6ffff0, 0.9);
        this.playerAuraGraphics.strokeCircle(
          player.position.x,
          player.position.y,
          Math.max(16, width * 0.026),
        );
        if (this.playerSprite) {
          this.playerSprite.setPosition(player.position.x, player.position.y);
          if (direction < 0) this.playerSprite.play("room2game-walk-left", true);
          else if (direction > 0) this.playerSprite.play("room2game-walk-right", true);
          else this.playerSprite.play("room2game-idle", true);
        } else {
          this.playerGraphics.fillStyle(0x5d9c67, 1);
          this.playerGraphics.fillRoundedRect(
            player.position.x - Math.max(7, width * 0.01),
            player.position.y - Math.max(10, height * 0.025),
            Math.max(14, width * 0.02),
            Math.max(20, height * 0.05),
            Math.max(5, width * 0.006),
          );
          this.playerGraphics.fillStyle(0xe6b887, 1);
          this.playerGraphics.fillCircle(
            player.position.x,
            player.position.y - Math.max(12, height * 0.028),
            Math.max(6, width * 0.009),
          );
        }
        for (const [objectIndex, sprite] of this.spriteImages) {
          const body = this.objectBodies.get(objectIndex);
          if (!body) continue;
          const offset = this.spriteOffsets?.get(objectIndex) || { x: 0, y: 0 };
          sprite.setPosition(body.position.x + offset.x, body.position.y + offset.y);
        }
        this.playerLight.setPosition(player.position.x, player.position.y);
        this.drawShadows();
      }
    }

    const game = new Phaser.Game({
      type: Phaser.WEBGL,
      parent: gameRoot,
      width,
      height,
      backgroundColor: "#171b14",
      render: { antialias: true, pixelArt: false, roundPixels: false },
      scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH,
        width,
        height,
      },
      physics: {
        default: "matter",
        matter: {
          gravity: { x: 0, y: 1.05 },
          debug: false,
          enableSleeping: false,
        },
      },
      scene: [RoomScene],
    });

    return {
      game,
      reset() {
        const scene = game.scene.getScene("Room2GameScene");
        if (scene && scene.scene.isActive()) scene.resetGame();
      },
      destroy() {
        game.destroy(true);
        collected.clear();
        collectiblesById.clear();
        score = 0;
      },
    };
  }

  window.Room2Game = { createGame };

  const photoInput = document.getElementById("photo-input");
  if (!photoInput) return;

  const dropzone = document.getElementById("dropzone");
  const photoPreview = document.getElementById("photo-preview");
  const uploadTitle = document.getElementById("upload-title");
  const uploadHint = document.getElementById("upload-hint");
  const generateButton = document.getElementById("generate-button");
  const playAgainButton = document.getElementById("play-again-button");
  const statusElement = document.getElementById("status");
  const gameMeta = document.getElementById("game-meta");
  const themeSelect = document.getElementById("theme");

  let selectedPhoto = null;
  let previewUrl = null;
  let activeGame = null;
  let lastScene = null;

  function setStatus(message, state) {
    statusElement.textContent = message;
    statusElement.dataset.state = state || "idle";
  }

  function selectPhoto(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setStatus("Choose a supported image file.", "error");
      return;
    }
    selectedPhoto = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    photoPreview.src = previewUrl;
    photoPreview.hidden = false;
    dropzone.classList.add("has-image");
    uploadTitle.textContent = file.name;
    uploadHint.textContent = `${Math.max(1, Math.round(file.size / 1024))} KB · click to change`;
    generateButton.disabled = false;
    setStatus("Ready to turn this photo into a playable scene.", "idle");
  }

  photoInput.addEventListener("change", (event) => {
    selectPhoto(event.target.files && event.target.files[0]);
  });

  for (const eventName of ["dragenter", "dragover"]) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-dragging");
    });
  }
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) selectPhoto(file);
  });

  generateButton.addEventListener("click", async () => {
    if (!selectedPhoto) return;
    generateButton.disabled = true;
    playAgainButton.disabled = true;
    setStatus("Analyzing the room and building its geometry…", "idle");
    gameMeta.textContent = "Generating scene";

    const formData = new FormData();
    formData.append("image", selectedPhoto, selectedPhoto.name);
    const apiOrigin = window.location.port === "8000" ? "" : "http://localhost:8000";
    try {
      const response = await fetch(`${apiOrigin}/generate?theme=${encodeURIComponent(themeSelect.value)}`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail = payload && payload.detail ? payload.detail : `Request failed (${response.status}).`;
        throw new Error(detail);
      }
      assertSceneData(payload);
      if (activeGame) activeGame.destroy();
      lastScene = payload;
      activeGame = createGame(payload, {});
      const objectCount = payload.objects.filter((object) => object.label !== "ground").length;
      gameMeta.textContent = `${objectCount} object${objectCount === 1 ? "" : "s"} · ${payload.canvas.width} × ${payload.canvas.height}`;
      setStatus("Scene ready. Find the objects, collect points, and avoid hazards.", "success");
      playAgainButton.disabled = false;
    } catch (error) {
      console.error("Room2Game generation failed", error);
      gameMeta.textContent = "Generation failed";
      setStatus(error instanceof Error ? error.message : "The scene could not be generated.", "error");
    } finally {
      generateButton.disabled = !selectedPhoto;
    }
  });

  playAgainButton.addEventListener("click", () => {
    if (!activeGame || !lastScene) return;
    activeGame.reset();
    setStatus("Level reset — have another run.", "success");
  });
})();
