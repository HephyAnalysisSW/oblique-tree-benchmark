<!doctype html>
<html><head><meta charset="utf-8"><title>Tree benchmark figures</title></head>
<body><h2>Tree benchmark figures</h2><ul>
<?php
// List the current flat layout; older URLs remain accessible directly.
$groups = array_merge(glob('tao_*', GLOB_ONLYDIR), glob('xgboost_*', GLOB_ONLYDIR), array('comparisons', 'diagnostics'));
sort($groups);
foreach ($groups as $group) {
    if (is_dir($group)) {
        $name = htmlspecialchars($group, ENT_QUOTES, 'UTF-8');
        echo '<li><a href="' . $name . '/">' . $name . '</a></li>';
    }
}
?>
</ul></body></html>
