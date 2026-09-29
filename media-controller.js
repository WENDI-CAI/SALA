/* Play every visible animation; release sources when they leave the viewport. */
(() => {
  "use strict";

  window.createCrystalPlayback = function createCrystalPlayback({ dialog, paused }) {
    const videos = [...document.querySelectorAll("video[data-playback]")];
    const states = new Map(videos.map(video => [video, {
      wanted: false, generation: 0, source: null, failedSource: null
    }]));
    let modalVideo = null;
    let frame = null;
    let pageAway = false;

    function release(video) {
      const state = states.get(video);
      if (!state.wanted && !video.hasAttribute("src")) return;
      state.wanted = false;
      state.generation += 1;
      video.pause();
      video.closest(".crystal-media").classList.remove("is-playing");
      if (video.hasAttribute("src")) {
        video.removeAttribute("src");
        // Reset the media element to release its decoder and buffered frames.
        video.load();
      }
    }

    function available(video) {
      return video.dataset.src && states.get(video).failedSource !== video.dataset.src;
    }

    function fail(video, source, generation) {
      const state = states.get(video);
      if (!state.wanted || state.generation !== generation || state.source !== source ||
          video.dataset.src !== source || video.getAttribute("src") !== source) return;
      // A blocked or broken source stays on its poster until the user retries.
      // Scrolling must not repeatedly download it or attempt autoplay again.
      state.failedSource = source;
      release(video);
      schedule();
    }

    function activate(video) {
      const state = states.get(video);
      if (state.wanted || !available(video)) return;
      const source = video.dataset.src;
      state.wanted = true;
      state.source = source;
      const generation = ++state.generation;
      video.muted = true;
      video.src = source;
      try {
        video.play()?.catch(() => fail(video, source, generation));
      } catch {
        fail(video, source, generation);
      }
    }

    function inViewport(video) {
      if (video.closest(".structure-card")?.hidden) return false;
      const bounds = video.closest(".crystal-media").getBoundingClientRect();
      if (bounds.width <= 0 || bounds.height <= 0) return false;
      const visibleWidth = Math.max(0, Math.min(bounds.right, window.innerWidth) - Math.max(bounds.left, 0));
      const visibleHeight = Math.max(0, Math.min(bounds.bottom, window.innerHeight) - Math.max(bounds.top, 74));
      return visibleWidth > 0 && visibleHeight > 0;
    }

    function refresh() {
      if (frame !== null) {
        window.cancelAnimationFrame(frame);
        frame = null;
      }
      let wanted = [];
      if (!paused() && !document.hidden && !pageAway) {
        if (dialog.open) {
          if (modalVideo && available(modalVideo) && !modalVideo.closest(".crystal-media").hidden) wanted = [modalVideo];
        } else {
          wanted = videos.filter(video => video.dataset.playback !== "detail" && available(video) && inViewport(video));
        }
      }
      const selected = new Set(wanted);
      // Stop old players before starting replacements, including on modal open.
      videos.forEach(video => { if (!selected.has(video)) release(video); });
      wanted.forEach(activate);
    }

    function schedule() {
      if (frame === null) frame = window.requestAnimationFrame(refresh);
    }

    videos.forEach(video => {
      video.addEventListener("playing", () => {
        const state = states.get(video);
        if (state.wanted && video.readyState >= 2 && video.currentSrc === video.src) {
          video.closest(".crystal-media").classList.add("is-playing");
        } else if (!state.wanted) video.pause();
      });
      video.addEventListener("error", () => {
        const state = states.get(video);
        // load() clears video.error. Ignore queued errors from an old source.
        if (!video.error || (video.currentSrc && video.currentSrc !== video.src)) return;
        fail(video, state.source, state.generation);
      });
    });

    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(schedule, { threshold: [0, 0.25, 0.5, 0.75, 1] });
      videos.filter(video => video.dataset.playback !== "detail")
        .forEach(video => observer.observe(video.closest(".crystal-media")));
    }
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule, { passive: true });
    document.addEventListener("visibilitychange", refresh);
    window.addEventListener("pagehide", () => { pageAway = true; refresh(); });
    window.addEventListener("pageshow", () => { pageAway = false; schedule(); });

    return {
      refresh,
      retry() {
        states.forEach(state => { state.failedSource = null; });
        refresh();
      },
      setSource(video, source) {
        if (video.dataset.src === source) return;
        release(video);
        video.dataset.src = source;
        states.get(video).failedSource = null;
      },
      setModal(video) {
        modalVideo = video;
        refresh();
      }
    };
  };
})();
