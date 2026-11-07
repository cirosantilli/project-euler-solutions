import solvers.«98.py.lean.d».Candidate

namespace solvers.«98».mapped_word
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.AnagramSquares
open pattern_signature (program)
open squares_with_length (squares)
open data.«project-euler-statements».data.py.project_euler_statements.p98.mapping (value table)
set_option Elab.async false

def Pending (env : HeapCalls.Env) (code : List Stmt) (heap : Mutable.Heap)
    (finalEnv : HeapCalls.Env) (rest : List Stmt) (finalHeap : Mutable.Heap) : Prop :=
  ∀ following anchors stack, Reaches program anchors
    ⟨.block (code ++ following) env, stack, heap⟩
    ⟨.block (rest ++ following) finalEnv, stack, finalHeap⟩

def score (width : Nat) (word digits : List Char) (other : String) (firstSquare best : Int) : Int :=
  match other.toList with
  | [] => best
  | first :: _ =>
      if substitution '0' word digits first = '0' then best else
        let text := String.ofList (other.toList.map (substitution '0' word digits))
        if text ∈ squares width then max (max best firstSquare) (integerValue text) else best

/-- One whole partner attempt either skips a leading zero or joins, tests, parses, and updates the maximum. -/
theorem result (width : Int) (word digits : List Char) (other : String) (firstSquare best : Int)
    (heap : Mutable.Heap) (mapping outer recordAddress : Nat) (env : HeapCalls.Env) (next : Stmt)
    (matched : labels [] word = labels [] digits) (nonempty : other.toList ≠ [])
    (included : ∀ letter ∈ other.toList, letter ∈ word)
    (found : heap[mapping]? = some (.dict (table (alphabet [] word) (alphabet [] digits))))
    (stored : square_cache.Stored outer heap width recordAddress)
    (getMapping : env "char_to_digit" = some (.ref mapping)) (getOther : env "w2" = some (.scalar (.string other)))
    (getRecord : env "sqinfo" = some (.ref recordAddress)) (getFirst : env "sq1" = some (number firstSquare))
    (getBest : env "best" = some (number best)) :
    ∃ finalEnv extra, Pending env (body ++ [.loopResume next]) heap finalEnv [next] (heap ++ extra) ∧
      finalEnv "best" = some (number (score width.toNat word digits other firstSquare best)) ∧
      (∀ name, name ∉ ["num2", "sq2", "best"] → finalEnv name = env name) := by
  obtain ⟨first, rest, view⟩ := List.exists_cons_of_ne_nil nonempty
  have leading := leading_zero_result word digits other first rest heap mapping env [] matched
    (included first (by simp [view])) view found getMapping getOther
  by_cases zero : substitution '0' word digits first = '0'
  · refine ⟨env, [], ?_, ?_, by intros; rfl⟩
    · intro following anchors stack
      have resumed : Reaches program anchors
          ⟨.block (.continueLoop :: body.drop 1 ++ (.loopResume next :: following)) env, stack, heap⟩
          ⟨.block (next :: following) env, stack, heap⟩ := .next rfl .refl
      simp only [List.append_nil]
      apply Reaches.next rfl
      apply (leading anchors _).trans
      refine Reaches.next ?_ resumed
      simp [step, run, Mutable.truth, scalarTruth, zero]
      rfl
    · simpa only [score, view, if_pos zero] using getBest
  · let text := String.ofList (other.toList.map (substitution '0' word digits))
    let output := Mutable.Object.list ((other.toList.map (substitution '0' word digits)).map value)
    let ready := setLocal env "num2" (.scalar (.string text))
    have guard : BlockContinues program env [.ifThen leadingZero [.continueLoop] []] heap env heap :=
      BlockContinues.branch (flag := false) leading (by simp [Mutable.truth, scalarTruth, zero]) BlockContinues.nil
    have assigned : BlockContinues program env [.assign [.name "num2"] joined] heap ready (heap ++ [output]) :=
      BlockContinues.assign (joined_result word digits other heap mapping env [] matched included found getMapping getOther)
    obtain ⟨finalEnv, updated, answer, kept⟩ := update_result width outer recordAddress text firstSquare best
      (heap ++ [output]) ready (stored.append [output])
      (by simpa [ready, setLocal] using getRecord) (by simp [ready, setLocal])
      (by simpa [ready, setLocal] using getFirst) (by simpa [ready, setLocal] using getBest)
    have executed : BlockContinues program env body heap finalEnv (heap ++ [output]) := guard.append (assigned.append updated)
    refine ⟨finalEnv, [output], ?_, ?_, ?_⟩
    · intro following anchors stack
      exact (executed (.loopResume next :: following) anchors stack).trans (.next rfl .refl)
    · simpa only [score, view, if_neg zero, text, view] using answer
    · intro name absent
      have notOutput : name ≠ "num2" := fun same => absent (by simp [same])
      rw [kept name (fun member => absent (List.mem_cons_of_mem _ member))]
      simp [ready, setLocal, notOutput]

end solvers.«98».mapped_word
