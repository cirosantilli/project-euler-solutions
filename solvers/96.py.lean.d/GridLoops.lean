import solvers.«96.py.lean.d».GridCell
import ProjectEulerSolutions.Lib.HeapCollection

namespace solvers.«96».grid_to_cands
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku
set_option Elab.async false

theorem columns (base : Mutable.Heap) (board : Board) (r : Fin 9) (c remaining : Nat)
    (finish : c + remaining = 9) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getR : env "r" = some (number r.val)) (getRow : env "row" = some (text (Encoding.row board r)))
    (getCandidates : env "cands" = some (.ref (base.length + 1))) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "c" c 9 1 columnBody []]
      (work base board (r.val * 9 + c)) finalEnv (work base board (r.val * 9 + 9)) globals ∧
      (∀ name, name ∉ ["c", "ch", "d"] → finalEnv name = env name) := by
  induction remaining generalizing c env with
  | zero =>
      have atEnd : c = 9 := by omega
      subst c
      exact ⟨env, BlockContinues.range_done rfl .nil, by intros; rfl⟩
  | succ remaining ih =>
      have inside : c < 9 := by omega
      let entered := setLocal env "c" (number c)
      obtain ⟨afterColumn, scanned, kept⟩ := column base board r ⟨c, inside⟩ entered globals
        (by simpa [entered, setLocal] using getR) (by simpa [entered, setLocal] using getRow)
        (by simp [entered, setLocal]) (by simpa [entered, setLocal] using getCandidates)
      obtain ⟨finalEnv, rest, restKept⟩ := ih (c + 1) (by omega) afterColumn
        (by rw [kept "r" (by decide)]; simpa [entered, setLocal] using getR)
        (by rw [kept "row" (by decide)]; simpa [entered, setLocal] using getRow)
        (by rw [kept "cands" (by decide)]; simpa [entered, setLocal] using getCandidates)
      refine ⟨finalEnv, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, columnBody, fill,
          Bool.false_or, Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one, Nat.add_assoc]
          using scanned.append rest
      · intro name absent
        rw [restKept name absent, kept name (fun member => absent (List.mem_cons_of_mem _ member))]
        have notColumn : name ≠ "c" := fun same => absent (by simp [same])
        simp [entered, setLocal, notColumn]

theorem row (base : Mutable.Heap) (board : Board) (r : Fin 9) (address : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (grid : base[address]? = some (.list ((Encoding.rows board).map text)))
    (getGrid : env "grid_lines" = some (.ref address)) (getR : env "r" = some (number r.val))
    (getCandidates : env "cands" = some (.ref (base.length + 1))) :
    ∃ finalEnv, BlockContinues program env rowBody (work base board (r.val * 9))
      finalEnv (work base board (r.val * 9 + 9)) globals ∧
      (∀ name, name ∉ ["row", "c", "ch", "d"] → finalEnv name = env name) := by
  have fetched : Evaluates program env (.subscript (.name "grid_lines") (.name "r"))
      (work base board (r.val * 9)) (text (Encoding.row board r)) (work base board (r.val * 9)) globals := by
    apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getGrid) (Evaluates.lookup getR)
    have preserved := ProjectEulerSolutions.HeapCollection.prefix_found base
      [.list [number 511], .list ((Encoding.prepared board (r.val * 9)).map fun n : Nat => number n)] address _ grid
    change (work base board (r.val * 9))[address]? = _ at preserved
    have size : (Encoding.rows board).length = 9 := by simp [Encoding.rows]
    obtain ⟨_, selected⟩ := List.getElem_of_getElem? (Encoding.rows_get board r)
    have nonnegative : ¬ (r.val : Int) < 0 := by omega
    simp [Mutable.getItem, preserved, Mutable.index, size, not_le.mpr r.isLt, nonnegative, Except.map, selected]
  let ready := setLocal env "row" (text (Encoding.row board r))
  obtain ⟨finalEnv, scanned, kept⟩ := columns base board r 0 9 rfl ready globals
    (by simpa [ready, setLocal] using getR) (by simp [ready, setLocal])
    (by simpa [ready, setLocal] using getCandidates)
  refine ⟨finalEnv, ?_, ?_⟩
  · apply (BlockContinues.assign fetched).append
    apply BlockContinues.for_range (.cons Evaluates.constant .nil) rfl
    simpa only [Nat.add_zero] using scanned
  · intro name absent
    rw [kept name (fun member => absent (List.mem_cons_of_mem _ member))]
    have notRow : name ≠ "row" := fun same => absent (by simp [same])
    simp [ready, setLocal, notRow]

theorem rows (base : Mutable.Heap) (board : Board) (r remaining address : Nat)
    (finish : r + remaining = 9) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (grid : base[address]? = some (.list ((Encoding.rows board).map text)))
    (getGrid : env "grid_lines" = some (.ref address)) (getCandidates : env "cands" = some (.ref (base.length + 1))) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "r" r 9 1 rowBody []]
      (work base board (r * 9)) finalEnv (work base board 81) globals ∧
      (∀ name, name ∉ ["r", "row", "c", "ch", "d"] → finalEnv name = env name) := by
  induction remaining generalizing r env with
  | zero =>
      have atEnd : r = 9 := by omega
      subst r
      exact ⟨env, BlockContinues.range_done rfl .nil, by intros; rfl⟩
  | succ remaining ih =>
      have inside : r < 9 := by omega
      let entered := setLocal env "r" (number r)
      obtain ⟨afterRow, scanned, kept⟩ := row base board ⟨r, inside⟩ address entered globals grid
        (by simpa [entered, setLocal] using getGrid) (by simp [entered, setLocal])
        (by simpa [entered, setLocal] using getCandidates)
      obtain ⟨finalEnv, rest, restKept⟩ := ih (r + 1) (by omega) afterRow
        (by rw [kept "grid_lines" (by decide)]; simpa [entered, setLocal] using getGrid)
        (by rw [kept "cands" (by decide)]; simpa [entered, setLocal] using getCandidates)
      refine ⟨finalEnv, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simp only [Nat.add_mul, Nat.one_mul] at rest
        simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, rowBody, columnBody, fill,
          Bool.false_or, Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one]
          using scanned.append rest
      · intro name absent
        rw [restKept name absent, kept name (fun member => absent (List.mem_cons_of_mem _ member))]
        have notR : name ≠ "r" := fun same => absent (by simp [same])
        simp [entered, setLocal, notR]

end solvers.«96».grid_to_cands
