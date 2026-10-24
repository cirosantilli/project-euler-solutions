import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11351_small : List Nat := []
theorem row11351_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11351 row11351_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11351_block_005 : List Nat := []
theorem row11351_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11351 row11351_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11351_block_006 : List Nat := [11597]
def plan11351_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, (.firstFactor 13), .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .second, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first, .second, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 37), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row11351_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11351 row11351_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11351_block_006
  · decide
  · rfl

def row11351_block_007 : List Nat := [12659, 12821, 13451, 14087, 14411]
def plan11351_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 31), (.secondFactor 23), .possible, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .possible, .first, .first, .first, .first, .first, .second, (.firstFactor 17), .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 37), (.secondFactor 7), .first, .second, .first, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .possible, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 23), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .second, .first, .possible, .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .second, (.firstFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 23), .first, .first, (.secondFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row11351_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11351 row11351_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11351_block_007
  · decide
  · rfl

def row11351_block_008 : List Nat := []
def plan11351_block_008 : List Budget.Choice := [.first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), .second, .first, (.firstFactor 17), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11351_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11351 row11351_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11351_block_008
  · decide
  · rfl

def row11351_block_009 : List Nat := []
theorem row11351_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11351 row11351_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11351_block_010 : List Nat := []
theorem row11351_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11351 row11351_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11351_block_011 : List Nat := []
theorem row11351_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11351 row11351_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11351_block_012 : List Nat := []
theorem row11351_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11351 row11351_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11351_block_013 : List Nat := []
theorem row11351_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11351 row11351_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11351_large_left_left : List Nat := row11351_small ++ row11351_block_005
theorem row11351_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11351 row11351_large_left_left :=
  Budget.RowUpperBound.branch row11351_small_covers row11351_block_005_covers

def row11351_large_left_right_right : List Nat := row11351_block_007 ++ row11351_block_008
theorem row11351_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11351 row11351_large_left_right_right :=
  Budget.RowUpperBound.branch row11351_block_007_covers row11351_block_008_covers

def row11351_large_left_right : List Nat := row11351_block_006 ++ row11351_large_left_right_right
theorem row11351_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11351 row11351_large_left_right :=
  Budget.RowUpperBound.branch row11351_block_006_covers row11351_large_left_right_right_covers

def row11351_large_left : List Nat := row11351_large_left_left ++ row11351_large_left_right
theorem row11351_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11351 row11351_large_left :=
  Budget.RowUpperBound.branch row11351_large_left_left_covers row11351_large_left_right_covers

def row11351_large_right_left : List Nat := row11351_block_009 ++ row11351_block_010
theorem row11351_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11351 row11351_large_right_left :=
  Budget.RowUpperBound.branch row11351_block_009_covers row11351_block_010_covers

def row11351_large_right_right_right : List Nat := row11351_block_012 ++ row11351_block_013
theorem row11351_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11351 row11351_large_right_right_right :=
  Budget.RowUpperBound.branch row11351_block_012_covers row11351_block_013_covers

def row11351_large_right_right : List Nat := row11351_block_011 ++ row11351_large_right_right_right
theorem row11351_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11351 row11351_large_right_right :=
  Budget.RowUpperBound.branch row11351_block_011_covers row11351_large_right_right_right_covers

def row11351_large_right : List Nat := row11351_large_right_left ++ row11351_large_right_right
theorem row11351_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11351 row11351_large_right :=
  Budget.RowUpperBound.branch row11351_large_right_left_covers row11351_large_right_right_covers

def row11351_large : List Nat := row11351_large_left ++ row11351_large_right
theorem row11351_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11351 row11351_large :=
  Budget.RowUpperBound.branch row11351_large_left_covers row11351_large_right_covers

def row11351 : List Nat := row11351_large
theorem row11351_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11351 row11351 := row11351_large_covers

def row11353_small : List Nat := []
theorem row11353_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11353 row11353_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11353_block_005 : List Nat := []
theorem row11353_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11353 row11353_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11353_block_006 : List Nat := [11719, 11941, 12289]
def plan11353_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .possible, .first, (.firstFactor 19), .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11353_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11353 row11353_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11353_block_006
  · decide
  · rfl

def row11353_block_007 : List Nat := [12433, 12601, 13687, 13789]
def plan11353_block_007 : List Budget.Choice := [(.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .second, .second, .first, .first, (.firstFactor 11), (.firstFactor 17), .second, .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 7), .possible, .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 13), .second, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .second, (.firstFactor 31), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .second, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, .possible, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 37), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11353_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11353 row11353_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11353_block_007
  · decide
  · rfl

def row11353_block_008 : List Nat := []
def plan11353_block_008 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11353_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11353 row11353_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11353_block_008
  · decide
  · rfl

def row11353_block_009 : List Nat := []
theorem row11353_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11353 row11353_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11353_block_010 : List Nat := []
theorem row11353_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11353 row11353_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11353_block_011 : List Nat := []
theorem row11353_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11353 row11353_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11353_block_012 : List Nat := []
theorem row11353_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11353 row11353_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11353_block_013 : List Nat := []
theorem row11353_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11353 row11353_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11353_large_left_left : List Nat := row11353_small ++ row11353_block_005
theorem row11353_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11353 row11353_large_left_left :=
  Budget.RowUpperBound.branch row11353_small_covers row11353_block_005_covers

def row11353_large_left_right_right : List Nat := row11353_block_007 ++ row11353_block_008
theorem row11353_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11353 row11353_large_left_right_right :=
  Budget.RowUpperBound.branch row11353_block_007_covers row11353_block_008_covers

def row11353_large_left_right : List Nat := row11353_block_006 ++ row11353_large_left_right_right
theorem row11353_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11353 row11353_large_left_right :=
  Budget.RowUpperBound.branch row11353_block_006_covers row11353_large_left_right_right_covers

def row11353_large_left : List Nat := row11353_large_left_left ++ row11353_large_left_right
theorem row11353_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11353 row11353_large_left :=
  Budget.RowUpperBound.branch row11353_large_left_left_covers row11353_large_left_right_covers

def row11353_large_right_left : List Nat := row11353_block_009 ++ row11353_block_010
theorem row11353_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11353 row11353_large_right_left :=
  Budget.RowUpperBound.branch row11353_block_009_covers row11353_block_010_covers

def row11353_large_right_right_right : List Nat := row11353_block_012 ++ row11353_block_013
theorem row11353_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11353 row11353_large_right_right_right :=
  Budget.RowUpperBound.branch row11353_block_012_covers row11353_block_013_covers

def row11353_large_right_right : List Nat := row11353_block_011 ++ row11353_large_right_right_right
theorem row11353_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11353 row11353_large_right_right :=
  Budget.RowUpperBound.branch row11353_block_011_covers row11353_large_right_right_right_covers

def row11353_large_right : List Nat := row11353_large_right_left ++ row11353_large_right_right
theorem row11353_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11353 row11353_large_right :=
  Budget.RowUpperBound.branch row11353_large_right_left_covers row11353_large_right_right_covers

def row11353_large : List Nat := row11353_large_left ++ row11353_large_right
theorem row11353_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11353 row11353_large :=
  Budget.RowUpperBound.branch row11353_large_left_covers row11353_large_right_covers

def row11353 : List Nat := row11353_large
theorem row11353_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11353 row11353 := row11353_large_covers

def row11369_small : List Nat := []
theorem row11369_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11369 row11369_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11369_block_005 : List Nat := []
theorem row11369_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11369 row11369_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11369_block_006 : List Nat := [11681, 12377]
def plan11369_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 13), .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .second, (.firstFactor 7), (.firstFactor 29), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .second, .first, .second, .first, .second, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11369_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11369 row11369_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11369_block_006
  · decide
  · rfl

def row11369_block_007 : List Nat := [13007, 13049, 13103, 14447]
def plan11369_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 37), .second, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 31), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 13), (.firstFactor 7), .possible, .first, .first, .first, .possible, (.secondFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 37), (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 31), .first, .second, .first, (.secondFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row11369_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11369 row11369_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11369_block_007
  · decide
  · rfl

def row11369_block_008 : List Nat := [14591, 14657]
def plan11369_block_008 : List Budget.Choice := [.first, (.secondFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .possible, .first, .second, (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 7), .first, .possible, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11369_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11369 row11369_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11369_block_008
  · decide
  · rfl

def row11369_block_009 : List Nat := []
theorem row11369_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11369 row11369_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11369_block_010 : List Nat := []
theorem row11369_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11369 row11369_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11369_block_011 : List Nat := []
theorem row11369_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11369 row11369_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11369_block_012 : List Nat := []
theorem row11369_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11369 row11369_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11369_block_013 : List Nat := []
theorem row11369_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11369 row11369_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11369_large_left_left : List Nat := row11369_small ++ row11369_block_005
theorem row11369_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11369 row11369_large_left_left :=
  Budget.RowUpperBound.branch row11369_small_covers row11369_block_005_covers

def row11369_large_left_right_right : List Nat := row11369_block_007 ++ row11369_block_008
theorem row11369_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11369 row11369_large_left_right_right :=
  Budget.RowUpperBound.branch row11369_block_007_covers row11369_block_008_covers

def row11369_large_left_right : List Nat := row11369_block_006 ++ row11369_large_left_right_right
theorem row11369_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11369 row11369_large_left_right :=
  Budget.RowUpperBound.branch row11369_block_006_covers row11369_large_left_right_right_covers

def row11369_large_left : List Nat := row11369_large_left_left ++ row11369_large_left_right
theorem row11369_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11369 row11369_large_left :=
  Budget.RowUpperBound.branch row11369_large_left_left_covers row11369_large_left_right_covers

def row11369_large_right_left : List Nat := row11369_block_009 ++ row11369_block_010
theorem row11369_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11369 row11369_large_right_left :=
  Budget.RowUpperBound.branch row11369_block_009_covers row11369_block_010_covers

def row11369_large_right_right_right : List Nat := row11369_block_012 ++ row11369_block_013
theorem row11369_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11369 row11369_large_right_right_right :=
  Budget.RowUpperBound.branch row11369_block_012_covers row11369_block_013_covers

def row11369_large_right_right : List Nat := row11369_block_011 ++ row11369_large_right_right_right
theorem row11369_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11369 row11369_large_right_right :=
  Budget.RowUpperBound.branch row11369_block_011_covers row11369_large_right_right_right_covers

def row11369_large_right : List Nat := row11369_large_right_left ++ row11369_large_right_right
theorem row11369_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11369 row11369_large_right :=
  Budget.RowUpperBound.branch row11369_large_right_left_covers row11369_large_right_right_covers

def row11369_large : List Nat := row11369_large_left ++ row11369_large_right
theorem row11369_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11369 row11369_large :=
  Budget.RowUpperBound.branch row11369_large_left_covers row11369_large_right_covers

def row11369 : List Nat := row11369_large
theorem row11369_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11369 row11369 := row11369_large_covers

def row11383_small : List Nat := []
theorem row11383_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11383 row11383_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11383_block_005 : List Nat := []
theorem row11383_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11383 row11383_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11383_block_006 : List Nat := [11467]
def plan11383_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 31), .second, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 31), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row11383_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11383 row11383_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11383_block_006
  · decide
  · rfl

def row11383_block_007 : List Nat := [12739, 14071, 14461]
def plan11383_block_007 : List Budget.Choice := [(.secondFactor 29), .first, .first, (.secondFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .second, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 17), (.firstFactor 29), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 19), .first, .second, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, (.secondFactor 31), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 37), .possible]
set_option maxRecDepth 2048 in
theorem row11383_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11383 row11383_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11383_block_007
  · decide
  · rfl

def row11383_block_008 : List Nat := []
def plan11383_block_008 : List Budget.Choice := [.first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11383_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11383 row11383_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11383_block_008
  · decide
  · rfl

def row11383_block_009 : List Nat := []
theorem row11383_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11383 row11383_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11383_block_010 : List Nat := []
theorem row11383_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11383 row11383_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11383_block_011 : List Nat := []
theorem row11383_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11383 row11383_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11383_block_012 : List Nat := []
theorem row11383_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11383 row11383_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11383_block_013 : List Nat := []
theorem row11383_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11383 row11383_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11383_large_left_left : List Nat := row11383_small ++ row11383_block_005
theorem row11383_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11383 row11383_large_left_left :=
  Budget.RowUpperBound.branch row11383_small_covers row11383_block_005_covers

def row11383_large_left_right_right : List Nat := row11383_block_007 ++ row11383_block_008
theorem row11383_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11383 row11383_large_left_right_right :=
  Budget.RowUpperBound.branch row11383_block_007_covers row11383_block_008_covers

def row11383_large_left_right : List Nat := row11383_block_006 ++ row11383_large_left_right_right
theorem row11383_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11383 row11383_large_left_right :=
  Budget.RowUpperBound.branch row11383_block_006_covers row11383_large_left_right_right_covers

def row11383_large_left : List Nat := row11383_large_left_left ++ row11383_large_left_right
theorem row11383_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11383 row11383_large_left :=
  Budget.RowUpperBound.branch row11383_large_left_left_covers row11383_large_left_right_covers

def row11383_large_right_left : List Nat := row11383_block_009 ++ row11383_block_010
theorem row11383_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11383 row11383_large_right_left :=
  Budget.RowUpperBound.branch row11383_block_009_covers row11383_block_010_covers

def row11383_large_right_right_right : List Nat := row11383_block_012 ++ row11383_block_013
theorem row11383_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11383 row11383_large_right_right_right :=
  Budget.RowUpperBound.branch row11383_block_012_covers row11383_block_013_covers

def row11383_large_right_right : List Nat := row11383_block_011 ++ row11383_large_right_right_right
theorem row11383_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11383 row11383_large_right_right :=
  Budget.RowUpperBound.branch row11383_block_011_covers row11383_large_right_right_right_covers

def row11383_large_right : List Nat := row11383_large_right_left ++ row11383_large_right_right
theorem row11383_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11383 row11383_large_right :=
  Budget.RowUpperBound.branch row11383_large_right_left_covers row11383_large_right_right_covers

def row11383_large : List Nat := row11383_large_left ++ row11383_large_right
theorem row11383_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11383 row11383_large :=
  Budget.RowUpperBound.branch row11383_large_left_covers row11383_large_right_covers

def row11383 : List Nat := row11383_large
theorem row11383_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11383 row11383 := row11383_large_covers

end ProjectEulerSolutions.PrimePairLarge
