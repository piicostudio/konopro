# PLAN.md (GSD Redesign)

This plan outlines the redesign of the KonoPro web MVP frontend to align with the user journey map, shadcn UI aesthetics, Vanta.js animated background, and GSAP animations.

<objective>
Redesign the KonoPro web MVP UI into a split-pane workspace (left inputs, right live results/status) on desktop, and a single vertical flow on mobile.
Implement Vanta.js network background, GSAP entrance/transition animations, and Shadcn-inspired modern dark mode elements.
</objective>

<process>

<task type="auto" effort="medium">
  <name>Integrate CDN libraries for Vanta.js and GSAP</name>
  <files>konopro/web/index.html</files>
  <action>
    Add Three.js, Vanta.net, and GSAP CDN scripts to the head/bottom of index.html.
    Remove legacy style.css if needed, or replace it with a clean new index.css.
  </action>
  <verify>
    Verify scripts load and no JavaScript console errors are thrown when opening the index.html page.
  </verify>
  <done>
    CDN links for Three.js (r134), Vanta.net, and GSAP (3.12.5) are present in index.html.
  </done>
</task>

<task type="auto" effort="high">
  <name>Redesign HTML Structure for User Journey</name>
  <files>konopro/web/index.html</files>
  <action>
    Modify the HTML body to structure the user journey:
    - Background container for Vanta.js
    - Modern minimal header with KONO PRO brand logo and Settings button
    - Main container with split-pane layout:
      - Left column: Input section (YouTube URL, cover recording upload card, Analyze button)
      - Right column: Result/Status section (empty state, compact pipeline loading progress, score + coach feedback cards)
    - Bottom/popup: Email capture waitlist section with premium look.
    Remove survey questions from the active blocking pipeline; instead, place them or hide them to focus on the upload-to-result journey.
  </action>
  <verify>
    Verify DOM nodes map correctly to script bindings.
  </verify>
  <done>
    index.html contains the new split-pane structure with updated IDs matching the simplified flow.
  </done>
</task>

<task type="auto" effort="high">
  <name>Create Premium Shadcn-Style CSS Theme</name>
  <files>konopro/web/templatemo-frost-style.css</files>
  <action>
    Overwrite templatecss with a premium, responsive CSS layout.
    - Dark mode first (slate-900 / zinc-950 backdrop)
    - Clean borders (border-zinc-800), sleek inputs, high-contrast text (zinc-50 / zinc-400)
    - Neon violet/indigo accents (#8b5cf6 / #6366f1) matching shadcn style
    - Glassmorphism effects for cards (backdrop-filter: blur(12px) with border/bg opacity)
    - Flex/Grid split layout: left input, right result.
    - Responsiveness: Media query stack vertical layout on mobile.
    - Custom stylings for audio upload zones, players, result grids, badge markers, and buttons.
  </action>
  <verify>
    Inspect layout styling for clean typography, colors, padding, and responsive stacking.
  </verify>
  <done>
    templatemo-frost-style.css is overwritten with the premium dark-mode styling.
  </done>
</task>

<task type="auto" effort="high">
  <name>Refactor Client-Side JS with Vanta, GSAP, and Split Flow</name>
  <files>konopro/web/templatemo-frost-script.js</files>
  <action>
    Rewrite templatejs script logic:
    - Initialize Vanta.NET on `#vanta-bg` with custom violet/dark colors.
    - Setup GSAP timeline animations for landing load (fade up headline, inputs, result container).
    - Adapt form submission to handle the simplified user journey:
      - Validation of YouTube URL and take upload.
      - Transition from Empty state -> Compact Loading (via GSAP fade out/fade in).
      - Poll FastAPI backend.
      - Transition from Loading -> Complete showing overall score, main issues, problem moments with A/B audio clips.
      - Automatically trigger email waitlist card focus or display at the end of the analysis.
    - Keep analytics logs and session feedback intact.
  </action>
  <verify>
    Verify page loads, animations run, and form transitions work properly.
  </verify>
  <done>
    templatemo-frost-script.js is updated, Vanta/GSAP work, and scoring pipeline functions.
  </done>
</task>

</process>
