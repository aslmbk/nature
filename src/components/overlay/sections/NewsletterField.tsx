"use client";

/**
 * Footer sign-up field. Deliberately not a <form>: there is no endpoint, the value is
 * never read, nothing is sent and nothing ends up in the URL (a form without JS would
 * GET-submit the address into the query string). The button only shows a note.
 */
import { useId, useState } from "react";
import { ArrowIcon, ui as u } from "../ui";
import s from "./FinaleSection.module.css";

const IDLE = "One short letter each season. Nothing else.";
const DONE = "Thank you. This is a design preview, so nothing was sent.";

export default function NewsletterField() {
  const id = useId();
  const [done, setDone] = useState(false);
  return (
    <div className={s.field}>
      <label htmlFor={id} className={s.fieldLabel}>
        Season notes
      </label>
      <div className={s.inputWrap}>
        <input
          id={id}
          type="email"
          inputMode="email"
          autoComplete="off"
          placeholder="Your email"
          className={s.input}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              setDone(true);
            }
          }}
        />
        <button type="button" className={s.send} aria-label="Sign up for season notes" onClick={() => setDone(true)}>
          <ArrowIcon />
        </button>
      </div>
      <p className={s.fieldNote} aria-live="polite">
        {done ? DONE : IDLE}
      </p>
      <span className={u.srOnly}>Preview only: the address is not stored or sent.</span>
    </div>
  );
}
