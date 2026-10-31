import ProjectEulerSolutions.Lib.DigitDiceSelections
import solvers.«90.py.lean.d».Rotation

/-! Construction of each die's mask from the tuple yielded by combinations. -/
namespace solvers.«90».masks
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions DigitDice
set_option Elab.async false

def code : List Stmt :=
  match findFunctionIn program.body "solve" with
  | some (_, body) => executableBody body
  | none => []

def combinationCode : List Stmt :=
  match (code[2]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

def digitCode : List Stmt :=
  match (combinationCode[1]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

/-- The source mask accumulator handles any finite digit sequence, including repetitions. -/
theorem digit_loop (digits remaining : List Nat) (position accumulator : Nat)
    (source : SequenceSource) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (env : HeapCalls.Env)
    (sequence : Mutable.sequence (sequenceValue source) heap = some (.ok (digits.map number), heap))
    (suffix : digits.drop position = remaining) (getM : env "m" = some (number accumulator)) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "d" source position digitCode []] heap finalEnv heap globals ∧
      finalEnv "m" = some (number (accumulator ||| DigitMask.mask remaining)) ∧
      (∀ name, name ≠ "m" → name ≠ "d" → finalEnv name = env name) := by
  induction remaining generalizing position accumulator env with
  | nil =>
      refine ⟨env, ?_, ?_, by intros; rfl⟩
      · apply BlockContinues.sequence_done sequence
        · have missing : digits[position]? = none :=
            List.getElem?_eq_none_iff.mpr (List.drop_eq_nil_iff.mp suffix)
          simp [missing]
        · exact BlockContinues.nil
      · simpa [DigitMask.mask] using getM
  | cons digit rest ih =>
      have item : digits[position]? = some digit := by
        simpa [suffix] using (List.head?_drop (l := digits) (i := position)).symm
      have tail : digits.drop (position + 1) = rest := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      let bound := setLocal env "d" (number digit)
      let updated := setLocal bound "m" (number (accumulator ||| 2 ^ digit))
      have power : Evaluates program bound (.binOp (.constant (.int 1)) .leftShift (.name "d"))
          heap (number (2 ^ digit)) heap globals := by
        apply Evaluates.binary Evaluates.constant (Evaluates.lookup rfl)
        simp [Mutable.binary, scalarShiftLeft, integerShiftLeft, number,
          show ¬ (digit : Int) < 0 by omega]
      have update : BlockContinues program bound digitCode heap updated heap globals := by
        apply BlockContinues.augAssign (left := number accumulator) (right := number (2 ^ digit))
        · simpa [bound, setLocal] using getM
        · exact power
        · rfl
      obtain ⟨finalEnv, executed, result, preserved⟩ :=
        ih (position + 1) (accumulator ||| 2 ^ digit) updated tail rfl
      refine ⟨finalEnv, ?_, ?_, ?_⟩
      · apply BlockContinues.sequence_step (value := number digit) sequence (by simp [item])
        exact update.append executed
      · simpa only [DigitMask.mask, Nat.or_assoc] using result
      · intro name notM notD
        rw [preserved name notM notD]
        simp [updated, bound, setLocal, notM, notD]

/-- itertools' positional rows and the mathematical choices use the same complete ordering. -/
theorem choices_rows (count : Nat) (pool : List α) : choices count pool = Combinations.rows count pool := by
  exact DigitDice.choices_rows count pool

end solvers.«90».masks
