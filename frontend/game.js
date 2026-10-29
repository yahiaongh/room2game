(function () {
  "use strict";

  const CATEGORY = {
    player: 0x0001,
    platform: 0x0002,
    hazard: 0x0004,
    collectible: 0x0008,
  };
  const ROLE_COLORS = {
    platform: 0x83a766,
    hazard: 0xe8755e,
    collectible: 0xe7c85f,
  };
  const PLATFORM_COLORS = [
    0x80a56a,
    0x7993a2,
    0xb38b66,
    0x8d80a7,
    0x9ca45f,
  ];

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

  function centerAndLocalVertices(vertices) {
    if (!Array.isArray(vertices) || vertices.length < 3) {
      throw new TypeError("A physics polygon needs at least three vertices.");
    }
    const center = vertices.reduce(
      (sum, point) => ({ x: sum.x + point.x / vertices.length, y: sum.y + point.y / vertices.length }),
      { x: 0, y: 0 },
    );
    return {
      center,
      local: vertices.map((point) => [point.x - center.x, point.y - center.y]),
    };
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
    const savedPlayerStart = {
      x: Phaser.Math.Clamp(sceneData.rules.player_start.x, 0, width),
      y: Phaser.Math.Clamp(sceneData.rules.player_start.y, 0, height),
    };

    class RoomScene extends Phaser.Scene {
      constructor() {
        super({ key: "Room2GameScene" });
      }

      create() {
        this.matter.world.setBounds(0, 0, width, height, 64, true, true, false, true);
        this.matter.world.setGravity(0, 1.05);
        this.cameras.main.setBackgroundColor("#171b14");

        this.backgroundGraphics = this.add.graphics().setDepth(0);
        this.backgroundGraphics.fillGradientStyle(
          0x293324,
          0x293324,
          0x171b14,
          0x171b14,
          1,
        );
        this.backgroundGraphics.fillRect(0, 0, width, height);
        this.backgroundGraphics.lineStyle(1, 0xc3d79a, 0.045);
        const gridGap = Math.max(32, Math.round(Math.min(width, height) / 12));
        for (let x = gridGap; x < width; x += gridGap) {
          this.backgroundGraphics.lineBetween(x, 0, x, height);
        }
        for (let y = gridGap; y < height; y += gridGap) {
          this.backgroundGraphics.lineBetween(0, y, width, y);
        }

        this.levelGraphics = this.add.graphics().setDepth(2);
        this.playerGraphics = this.add.graphics().setDepth(4);
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

        this.playerStart = savedPlayerStart;
        this.playerBody = this.matter.add.rectangle(
          this.playerStart.x,
          this.playerStart.y,
          Math.max(18, width * 0.025),
          Math.max(24, height * 0.06),
          {
            label: "room2game-player",
            friction: 0.08,
            frictionAir: 0.018,
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

        this.keys = this.input.keyboard.addKeys({
          left: Phaser.Input.Keyboard.KeyCodes.LEFT,
          right: Phaser.Input.Keyboard.KeyCodes.RIGHT,
          up: Phaser.Input.Keyboard.KeyCodes.UP,
          a: Phaser.Input.Keyboard.KeyCodes.A,
          d: Phaser.Input.Keyboard.KeyCodes.D,
          space: Phaser.Input.Keyboard.KeyCodes.SPACE,
        });
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
      }

      drawObjects(objects, role) {
        const roleEntries = sceneData.rules[`${role}s`];
        roleEntries.forEach((entry, order) => {
          const object = objects[entry.object_index];
          const points = object.display;
          if (!Array.isArray(points) || points.length < 3) return;
          const color = role === "platform"
            ? PLATFORM_COLORS[order % PLATFORM_COLORS.length]
            : ROLE_COLORS[role];
          this.levelGraphics.fillStyle(color, role === "platform" ? 0.94 : 0.9);
          this.levelGraphics.lineStyle(2, 0xf0f1e8, 0.22);
          this.levelGraphics.fillPoints(points, true);
          this.levelGraphics.strokePoints(points, true);
          if (role === "platform") {
            this.levelGraphics.lineStyle(Math.max(2, height * 0.008), 0xe0efc3, 0.33);
            this.levelGraphics.beginPath();
            this.levelGraphics.moveTo(points[0].x, points[0].y);
            this.levelGraphics.lineTo(points[1].x, points[1].y);
            this.levelGraphics.strokePath();
          }
        });
      }

      drawOptionalAssets(assetMap) {
        for (const object of sceneData.objects) {
          const key = assetMap[object.object_index];
          if (!key || !this.textures.exists(key) || !Array.isArray(object.bbox)) continue;
          const [left, top, right, bottom] = object.bbox;
          this.add.image((left + right) / 2, (top + bottom) / 2, key)
            .setDisplaySize(Math.max(1, right - left), Math.max(1, bottom - top))
            .setDepth(3);
        }
      }

      createRoleBodies(entries, role) {
        entries.forEach((entry) => {
          const object = sceneData.objects[entry.object_index];
          const { center, local } = centerAndLocalVertices(object.physics);
          const category = CATEGORY[role];
          const body = this.matter.add.fromVertices(
            center.x,
            center.y,
            [local],
            {
              label: `room2game-${role}-${entry.object_index}`,
              isStatic: role === "platform",
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
        if (scoreElement) scoreElement.textContent = `${score} point${score === 1 ? "" : "s"}`;
        this.drawCollectedState();
      }

      drawCollectedState() {
        this.levelGraphics.clear();
        this.drawObjects(sceneData.objects, "platform");
        this.drawObjects(sceneData.objects, "hazard");
        const collectedIndices = new Set(
          [...collected].map((id) => collectiblesById.get(id).object_index),
        );
        const entries = sceneData.rules.collectibles;
        const keepEntries = entries.filter((entry) => !collectedIndices.has(entry.object_index));
        this.drawObjectsForList(keepEntries, ROLE_COLORS.collectible);
      }

      drawObjectsForList(entries, color) {
        for (const entry of entries) {
          const object = sceneData.objects[entry.object_index];
          if (!object || !Array.isArray(object.display)) continue;
          this.levelGraphics.fillStyle(color, 0.9);
          this.levelGraphics.lineStyle(2, 0xf0f1e8, 0.22);
          this.levelGraphics.fillPoints(object.display, true);
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
        this.resetAfterHazard = false;
        this.canJump = false;
      }

      resetGame() {
        for (const bodyId of collected) {
          const body = this.bodiesById.get(bodyId);
          if (body) this.matter.world.add(body);
          this.collectibleBodies.add(bodyId);
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

      update() {
        const moveLeft = this.keys.left.isDown || this.keys.a.isDown;
        const moveRight = this.keys.right.isDown || this.keys.d.isDown;
        const direction = Number(moveRight) - Number(moveLeft);
        const maxSpeed = Math.max(3, width * 0.009);
        if (direction !== 0) {
          this.matter.body.setVelocity(this.playerBody, {
            x: direction * maxSpeed,
            y: this.playerBody.velocity.y,
          });
        } else if (Math.abs(this.playerBody.velocity.x) > 0.2) {
          this.matter.body.setVelocity(this.playerBody, {
            x: this.playerBody.velocity.x * 0.78,
            y: this.playerBody.velocity.y,
          });
        }

        if (
          Phaser.Input.Keyboard.JustDown(this.keys.space)
          || Phaser.Input.Keyboard.JustDown(this.keys.up)
        ) {
          this.jump();
        }
        if (this.resetAfterHazard) this.resetPlayer();
        if (this.playerBody.position.y > height + 80) this.resetPlayer();

        const player = this.playerBody;
        this.playerGraphics.clear();
        this.playerGraphics.fillStyle(0xd3f36b, 1);
        this.playerGraphics.fillRoundedRect(
          player.position.x - Math.max(9, width * 0.0125),
          player.position.y - Math.max(12, height * 0.03),
          Math.max(18, width * 0.025),
          Math.max(24, height * 0.06),
          Math.max(3, width * 0.006),
        );
        this.playerGraphics.fillStyle(0x252b1c, 1);
        this.playerGraphics.fillCircle(
          player.position.x + Math.max(3, width * 0.005),
          player.position.y - Math.max(4, height * 0.009),
          Math.max(1.5, width * 0.002),
        );
      }
    }

    const game = new Phaser.Game({
      type: Phaser.CANVAS,
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
          enableSleeping: true,
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
