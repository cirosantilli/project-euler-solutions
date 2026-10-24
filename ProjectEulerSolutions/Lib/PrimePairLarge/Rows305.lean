import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9923_small : List Nat := []
theorem row9923_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9923 row9923_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9923_block_005 : List Nat := []
def plan9923_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .second, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 7), .second, .second, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9923_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9923 row9923_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9923_block_005
  · decide
  · rfl

def row9923_block_006 : List Nat := [11393, 11447, 11699, 11783]
def plan9923_block_006 : List Budget.Choice := [.first, .first, (.secondFactor 11), .first, .second, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 17), .first, .first, (.secondFactor 19), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .possible, (.firstFactor 37), (.firstFactor 7), (.firstFactor 19), .first, .first, .possible, .first, (.firstFactor 17), (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, (.secondFactor 11), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .possible, (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 13), (.secondFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 23), .first, (.secondFactor 13), (.secondFactor 17), .first, (.firstFactor 29), (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 13), .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 29)]
set_option maxRecDepth 2048 in
theorem row9923_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9923 row9923_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9923_block_006
  · decide
  · rfl

def row9923_block_007 : List Nat := [12911, 14249]
def plan9923_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 13), (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .possible, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 29), (.secondFactor 17), (.firstFactor 13), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), (.secondFactor 7), .first, .second, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 23), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 17), .possible, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 23), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9923_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9923 row9923_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9923_block_007
  · decide
  · rfl

def row9923_block_008 : List Nat := []
def plan9923_block_008 : List Budget.Choice := [.first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .second, .first, .second, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .second, .first, .first, (.firstFactor 11), .second, (.secondFactor 17), (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 19), .second, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .second, .first, .first, .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9923_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9923 row9923_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9923_block_008
  · decide
  · rfl

def row9923_block_009 : List Nat := []
theorem row9923_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9923 row9923_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9923_block_010 : List Nat := []
theorem row9923_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9923 row9923_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9923_block_011 : List Nat := []
theorem row9923_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9923 row9923_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9923_block_012 : List Nat := []
theorem row9923_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9923 row9923_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9923_block_013 : List Nat := []
theorem row9923_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9923 row9923_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9923_large_left_left : List Nat := row9923_small ++ row9923_block_005
theorem row9923_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9923 row9923_large_left_left :=
  Budget.RowUpperBound.branch row9923_small_covers row9923_block_005_covers

def row9923_large_left_right_right : List Nat := row9923_block_007 ++ row9923_block_008
theorem row9923_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9923 row9923_large_left_right_right :=
  Budget.RowUpperBound.branch row9923_block_007_covers row9923_block_008_covers

def row9923_large_left_right : List Nat := row9923_block_006 ++ row9923_large_left_right_right
theorem row9923_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9923 row9923_large_left_right :=
  Budget.RowUpperBound.branch row9923_block_006_covers row9923_large_left_right_right_covers

def row9923_large_left : List Nat := row9923_large_left_left ++ row9923_large_left_right
theorem row9923_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9923 row9923_large_left :=
  Budget.RowUpperBound.branch row9923_large_left_left_covers row9923_large_left_right_covers

def row9923_large_right_left : List Nat := row9923_block_009 ++ row9923_block_010
theorem row9923_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9923 row9923_large_right_left :=
  Budget.RowUpperBound.branch row9923_block_009_covers row9923_block_010_covers

def row9923_large_right_right_right : List Nat := row9923_block_012 ++ row9923_block_013
theorem row9923_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9923 row9923_large_right_right_right :=
  Budget.RowUpperBound.branch row9923_block_012_covers row9923_block_013_covers

def row9923_large_right_right : List Nat := row9923_block_011 ++ row9923_large_right_right_right
theorem row9923_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9923 row9923_large_right_right :=
  Budget.RowUpperBound.branch row9923_block_011_covers row9923_large_right_right_right_covers

def row9923_large_right : List Nat := row9923_large_right_left ++ row9923_large_right_right
theorem row9923_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9923 row9923_large_right :=
  Budget.RowUpperBound.branch row9923_large_right_left_covers row9923_large_right_right_covers

def row9923_large : List Nat := row9923_large_left ++ row9923_large_right
theorem row9923_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9923 row9923_large :=
  Budget.RowUpperBound.branch row9923_large_left_covers row9923_large_right_covers

def row9923 : List Nat := row9923_large
theorem row9923_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9923 row9923 := row9923_large_covers

def row9929_small : List Nat := []
theorem row9929_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9929 row9929_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9929_block_005 : List Nat := []
def plan9929_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 19), .first, (.firstFactor 37), (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9929_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9929 row9929_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9929_block_005
  · decide
  · rfl

def row9929_block_006 : List Nat := [11069, 11261]
def plan9929_block_006 : List Budget.Choice := [(.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .second, .first, .first, .first, .second, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 23), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .possible, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 29), .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 31), .second, (.secondFactor 13), (.firstFactor 17), .first, (.secondFactor 23), .first, .first, (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row9929_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9929 row9929_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9929_block_006
  · decide
  · rfl

def row9929_block_007 : List Nat := [13331, 13877]
def plan9929_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 29), (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 31), (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), (.secondFactor 17), (.secondFactor 19), .second, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .second, .second, (.secondFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 17), (.firstFactor 19), (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .possible, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .second, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .second, .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9929_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9929 row9929_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9929_block_007
  · decide
  · rfl

def row9929_block_008 : List Nat := [14831, 15299]
def plan9929_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .possible, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .second, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9929_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9929 row9929_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9929_block_008
  · decide
  · rfl

def row9929_block_009 : List Nat := []
theorem row9929_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9929 row9929_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9929_block_010 : List Nat := []
theorem row9929_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9929 row9929_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9929_block_011 : List Nat := []
theorem row9929_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9929 row9929_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9929_block_012 : List Nat := []
theorem row9929_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9929 row9929_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9929_block_013 : List Nat := []
theorem row9929_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9929 row9929_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9929_large_left_left : List Nat := row9929_small ++ row9929_block_005
theorem row9929_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9929 row9929_large_left_left :=
  Budget.RowUpperBound.branch row9929_small_covers row9929_block_005_covers

def row9929_large_left_right_right : List Nat := row9929_block_007 ++ row9929_block_008
theorem row9929_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9929 row9929_large_left_right_right :=
  Budget.RowUpperBound.branch row9929_block_007_covers row9929_block_008_covers

def row9929_large_left_right : List Nat := row9929_block_006 ++ row9929_large_left_right_right
theorem row9929_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9929 row9929_large_left_right :=
  Budget.RowUpperBound.branch row9929_block_006_covers row9929_large_left_right_right_covers

def row9929_large_left : List Nat := row9929_large_left_left ++ row9929_large_left_right
theorem row9929_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9929 row9929_large_left :=
  Budget.RowUpperBound.branch row9929_large_left_left_covers row9929_large_left_right_covers

def row9929_large_right_left : List Nat := row9929_block_009 ++ row9929_block_010
theorem row9929_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9929 row9929_large_right_left :=
  Budget.RowUpperBound.branch row9929_block_009_covers row9929_block_010_covers

def row9929_large_right_right_right : List Nat := row9929_block_012 ++ row9929_block_013
theorem row9929_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9929 row9929_large_right_right_right :=
  Budget.RowUpperBound.branch row9929_block_012_covers row9929_block_013_covers

def row9929_large_right_right : List Nat := row9929_block_011 ++ row9929_large_right_right_right
theorem row9929_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9929 row9929_large_right_right :=
  Budget.RowUpperBound.branch row9929_block_011_covers row9929_large_right_right_right_covers

def row9929_large_right : List Nat := row9929_large_right_left ++ row9929_large_right_right
theorem row9929_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9929 row9929_large_right :=
  Budget.RowUpperBound.branch row9929_large_right_left_covers row9929_large_right_right_covers

def row9929_large : List Nat := row9929_large_left ++ row9929_large_right
theorem row9929_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9929 row9929_large :=
  Budget.RowUpperBound.branch row9929_large_left_covers row9929_large_right_covers

def row9929 : List Nat := row9929_large
theorem row9929_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9929 row9929 := row9929_large_covers

def row9931_small : List Nat := []
theorem row9931_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9931 row9931_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9931_block_005 : List Nat := []
def plan9931_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row9931_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9931 row9931_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9931_block_005
  · decide
  · rfl

def row9931_block_006 : List Nat := [10453, 10711, 11149, 11887, 11941, 12277]
def plan9931_block_006 : List Budget.Choice := [.first, (.firstFactor 23), .first, .second, .first, .possible, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, .possible, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 19), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 31), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .possible, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 31), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row9931_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9931 row9931_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9931_block_006
  · decide
  · rfl

def row9931_block_007 : List Nat := [12451, 12739, 13309, 13441, 13963]
def plan9931_block_007 : List Budget.Choice := [(.firstFactor 7), .second, .first, .possible, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .second, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .possible, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 37), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .second, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 19), .second, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row9931_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9931 row9931_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9931_block_007
  · decide
  · rfl

def row9931_block_008 : List Nat := [15913]
def plan9931_block_008 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 11), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 29), .first, .first, .second, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 17), (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .possible, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9931_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9931 row9931_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9931_block_008
  · decide
  · rfl

def row9931_block_009 : List Nat := []
theorem row9931_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9931 row9931_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9931_block_010 : List Nat := []
theorem row9931_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9931 row9931_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9931_block_011 : List Nat := []
theorem row9931_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9931 row9931_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9931_block_012 : List Nat := []
theorem row9931_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9931 row9931_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9931_block_013 : List Nat := []
theorem row9931_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9931 row9931_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9931_large_left_left : List Nat := row9931_small ++ row9931_block_005
theorem row9931_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9931 row9931_large_left_left :=
  Budget.RowUpperBound.branch row9931_small_covers row9931_block_005_covers

def row9931_large_left_right_right : List Nat := row9931_block_007 ++ row9931_block_008
theorem row9931_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9931 row9931_large_left_right_right :=
  Budget.RowUpperBound.branch row9931_block_007_covers row9931_block_008_covers

def row9931_large_left_right : List Nat := row9931_block_006 ++ row9931_large_left_right_right
theorem row9931_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9931 row9931_large_left_right :=
  Budget.RowUpperBound.branch row9931_block_006_covers row9931_large_left_right_right_covers

def row9931_large_left : List Nat := row9931_large_left_left ++ row9931_large_left_right
theorem row9931_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9931 row9931_large_left :=
  Budget.RowUpperBound.branch row9931_large_left_left_covers row9931_large_left_right_covers

def row9931_large_right_left : List Nat := row9931_block_009 ++ row9931_block_010
theorem row9931_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9931 row9931_large_right_left :=
  Budget.RowUpperBound.branch row9931_block_009_covers row9931_block_010_covers

def row9931_large_right_right_right : List Nat := row9931_block_012 ++ row9931_block_013
theorem row9931_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9931 row9931_large_right_right_right :=
  Budget.RowUpperBound.branch row9931_block_012_covers row9931_block_013_covers

def row9931_large_right_right : List Nat := row9931_block_011 ++ row9931_large_right_right_right
theorem row9931_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9931 row9931_large_right_right :=
  Budget.RowUpperBound.branch row9931_block_011_covers row9931_large_right_right_right_covers

def row9931_large_right : List Nat := row9931_large_right_left ++ row9931_large_right_right
theorem row9931_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9931 row9931_large_right :=
  Budget.RowUpperBound.branch row9931_large_right_left_covers row9931_large_right_right_covers

def row9931_large : List Nat := row9931_large_left ++ row9931_large_right
theorem row9931_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9931 row9931_large :=
  Budget.RowUpperBound.branch row9931_large_left_covers row9931_large_right_covers

def row9931 : List Nat := row9931_large
theorem row9931_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9931 row9931 := row9931_large_covers

def row9941_small : List Nat := []
theorem row9941_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9941 row9941_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9941_block_005 : List Nat := []
def plan9941_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), .first, .second, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row9941_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9941 row9941_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9941_block_005
  · decide
  · rfl

def row9941_block_006 : List Nat := [11579, 12107]
def plan9941_block_006 : List Budget.Choice := [.second, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 37), .second, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 31), .second, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 19), .first, (.secondFactor 19), .second, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), (.firstFactor 13), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .possible, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9941_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9941 row9941_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9941_block_006
  · decide
  · rfl

def row9941_block_007 : List Nat := [12473, 14207]
def plan9941_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .second, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 23), .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row9941_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9941 row9941_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9941_block_007
  · decide
  · rfl

def row9941_block_008 : List Nat := [15413, 15761, 15803]
def plan9941_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 29), .first, (.secondFactor 19), .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .second, .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 37), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, .possible, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 19), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .possible, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 31), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9941_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9941 row9941_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9941_block_008
  · decide
  · rfl

def row9941_block_009 : List Nat := []
theorem row9941_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9941 row9941_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9941_block_010 : List Nat := []
theorem row9941_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9941 row9941_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9941_block_011 : List Nat := []
theorem row9941_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9941 row9941_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9941_block_012 : List Nat := []
theorem row9941_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9941 row9941_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9941_block_013 : List Nat := []
theorem row9941_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9941 row9941_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9941_large_left_left : List Nat := row9941_small ++ row9941_block_005
theorem row9941_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9941 row9941_large_left_left :=
  Budget.RowUpperBound.branch row9941_small_covers row9941_block_005_covers

def row9941_large_left_right_right : List Nat := row9941_block_007 ++ row9941_block_008
theorem row9941_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9941 row9941_large_left_right_right :=
  Budget.RowUpperBound.branch row9941_block_007_covers row9941_block_008_covers

def row9941_large_left_right : List Nat := row9941_block_006 ++ row9941_large_left_right_right
theorem row9941_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9941 row9941_large_left_right :=
  Budget.RowUpperBound.branch row9941_block_006_covers row9941_large_left_right_right_covers

def row9941_large_left : List Nat := row9941_large_left_left ++ row9941_large_left_right
theorem row9941_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9941 row9941_large_left :=
  Budget.RowUpperBound.branch row9941_large_left_left_covers row9941_large_left_right_covers

def row9941_large_right_left : List Nat := row9941_block_009 ++ row9941_block_010
theorem row9941_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9941 row9941_large_right_left :=
  Budget.RowUpperBound.branch row9941_block_009_covers row9941_block_010_covers

def row9941_large_right_right_right : List Nat := row9941_block_012 ++ row9941_block_013
theorem row9941_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9941 row9941_large_right_right_right :=
  Budget.RowUpperBound.branch row9941_block_012_covers row9941_block_013_covers

def row9941_large_right_right : List Nat := row9941_block_011 ++ row9941_large_right_right_right
theorem row9941_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9941 row9941_large_right_right :=
  Budget.RowUpperBound.branch row9941_block_011_covers row9941_large_right_right_right_covers

def row9941_large_right : List Nat := row9941_large_right_left ++ row9941_large_right_right
theorem row9941_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9941 row9941_large_right :=
  Budget.RowUpperBound.branch row9941_large_right_left_covers row9941_large_right_right_covers

def row9941_large : List Nat := row9941_large_left ++ row9941_large_right
theorem row9941_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9941 row9941_large :=
  Budget.RowUpperBound.branch row9941_large_left_covers row9941_large_right_covers

def row9941 : List Nat := row9941_large
theorem row9941_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9941 row9941 := row9941_large_covers

end ProjectEulerSolutions.PrimePairLarge
