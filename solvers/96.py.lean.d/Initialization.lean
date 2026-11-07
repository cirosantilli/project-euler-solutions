import solvers.«96.py.lean.d».BuildTables

namespace solvers.«96».initialization
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

def code : List Stmt := program.body.filter fun statement => match statement with
  | .assign _ _ => true
  | _ => false

theorem source : code =
    [.assign [.name "ALL"] (.binOp (.binOp (.constant (.int 1)) .leftShift (.constant (.int 9))) .sub (.constant (.int 1))),
     .assign [.tuple [.name "PEERS", .name "UNITS"]] (.call (.name "build_peers_units") [])] := rfl

structure Context (caller : Mutable.Heap) where
  heap : Mutable.Heap
  tuple : Nat
  peers : PeerTable
  returned : ReturnsValue program "build_peers_units" [] caller (.ref tuple) heap []
  pair : heap[tuple]? = some (.tuple [.ref peers.root, .ref (build_peers_units.table caller.length).root])
  peerView : peers.View heap
  unitView : (build_peers_units.table caller.length).View heap
  length : caller.length ≤ heap.length
  frame : ∀ index < caller.length, heap[index]? = caller[index]?

/-- Choose only a heap and table pair witnessed by execution of the actual constructor. -/
def context (caller : Mutable.Heap) : Context caller := Classical.choice (by
  obtain ⟨heap, tuple, peers, returned, pair, peerView, unitView, length, frame⟩ := build_peers_units.result caller []
  exact ⟨⟨heap, tuple, peers, returned, pair, peerView, unitView, length, frame⟩⟩)

def env (caller : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal (setLocal (fun _ => none) "ALL" (number 511))
    "PEERS" (.ref (context caller).peers.root)) "UNITS" (.ref caller.length)

def globals (caller : Mutable.Heap) : Mutable.Globals :=
  [("UNITS", .ref caller.length), ("PEERS", .ref (context caller).peers.root), ("ALL", number 511)]

/-- Execute both module assignments, including the constructor call and tuple unpack. -/
theorem executes (caller : Mutable.Heap) :
    BlockContinues program (fun _ => none) code caller (env caller) (context caller).heap [] := by
  have allMask : Evaluates program (fun _ => none)
      (.binOp (.binOp (.constant (.int 1)) .leftShift (.constant (.int 9))) .sub (.constant (.int 1)))
      caller (number 511) caller [] := by
    apply Evaluates.binary (av := number 512) (bv := number 1) (middle := caller) (afterRight := caller)
    · exact Evaluates.binary Evaluates.constant Evaluates.constant rfl
    · exact Evaluates.constant
    · rfl
  rw [source]
  apply (BlockContinues.assign allMask).append
  apply BlockContinues.unpack (names := ["PEERS", "UNITS"])
    (items := [.ref (context caller).peers.root, .ref caller.length]) rfl
    (Evaluates.call .nil (context caller).returned)
    (by simp [Mutable.sequence, (context caller).pair, build_peers_units.table]) rfl

theorem exports (caller : Mutable.Heap) (name : String) : env caller name = (globals caller).lookup name := by
  by_cases units : name = "UNITS"
  · subst name
    rfl
  · by_cases peers : name = "PEERS"
    · subst name
      rfl
    · by_cases all : name = "ALL"
      · subst name
        rfl
      · simp [env, globals, setLocal, List.lookup, units, peers, all,
          beq_eq_false_iff_ne.mpr units, beq_eq_false_iff_ne.mpr peers, beq_eq_false_iff_ne.mpr all]

theorem complete (caller : Mutable.Heap) : (build_peers_units.table caller.length).Complete :=
  build_peers_units.table_complete caller.length

theorem get_all (caller : Mutable.Heap) : (globals caller).lookup "ALL" = some (number 511) := rfl
theorem get_peers (caller : Mutable.Heap) : (globals caller).lookup "PEERS" = some (.ref (context caller).peers.root) := rfl
theorem get_units (caller : Mutable.Heap) : (globals caller).lookup "UNITS" = some (.ref (build_peers_units.table caller.length).root) := rfl

end
end solvers.«96».initialization
