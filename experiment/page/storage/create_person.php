<?php

function generateRandomString($length = 10) {
    $characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    $charactersLength = strlen($characters);
    $randomString = '';
    for ($i = 0; $i < $length; $i++) {
        $randomString .= $characters[rand(0, $charactersLength - 1)];
    }
    return $randomString;
}

$vp_name = generateRandomString();
$max_it = 100;

while (file_exists($vp_name . ".csv")) {
	$vp_name = generateRandomString();
	
	if ($max_it <= 0) {
		echo json_encode(0);
		return;
	}
	$max_it = $max_it - 1;
}

$data = "Timestamp;Data";
$myfile = file_put_contents($vp_name . ".csv", $data.PHP_EOL , LOCK_EX);

echo json_encode($vp_name);

