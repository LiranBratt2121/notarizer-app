// C:\Users\bratt\Documents\code\notarizer-bubble\App.tsx

import React from 'react';
import {
  NativeModules, View, Text, Alert, Platform, StyleSheet,
  TouchableOpacity, SafeAreaView, StatusBar
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const { BubbleModule } = NativeModules;
const UPLOAD_URL = "http://10.0.0.102:8000/add-valid-image"; // Endpoint to add images

export default function App() {

  const handleStartBubble = async () => {
    try {
      await BubbleModule.startBubble();
    } catch (e: any) {
      if (e.code === 'NO_OVERLAY_PERMISSION') {
        Alert.alert('Permission Required', 'Please grant overlay permission, then try again.');
      } else if (e.code === 'SCREENSHOT_REJECTED') {
        Alert.alert('Permission Denied', 'You must grant screen capture permission.');
      } else {
        Alert.alert('Error', e.message);
      }
    }
  };

  const handleUploadValidImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (permissionResult.granted === false) {
      Alert.alert("Permission required", "You need to allow access to your photo library to upload images.");
      return;
    }

    const pickerResult = await ImagePicker.launchImageLibraryAsync();
    if (pickerResult.canceled === true) {
      return;
    }

    const uri = pickerResult.assets[0].uri;
    const formData = new FormData();
    formData.append('file', {
      uri: Platform.OS === 'android' ? uri : uri.replace('file://', ''),
      name: 'valid-image.jpg',
      type: 'image/jpeg',
    } as any);

    try {
      Alert.alert('Uploading...', 'Adding new valid image to database.');
      const response = await fetch(UPLOAD_URL, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const responseJson = await response.json();
      if (!response.ok) throw new Error(responseJson.detail || "Server error");
      Alert.alert('Success!', `Image hash added: ${responseJson.added_phash}`);
    } catch (error) {
      Alert.alert('Upload Failed', `Error: ${error}`);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" />
      <View style={styles.container}>
        <Text style={styles.title}>Notarizer Admin</Text>
        
        <TouchableOpacity style={styles.button} onPress={handleStartBubble}>
          <Text style={styles.buttonText}>Start Validation Bubble</Text>
        </TouchableOpacity>
        
        <Text style={styles.separatorText}>OR</Text>
        
        <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={handleUploadValidImage}>
          <Text style={[styles.buttonText, styles.secondaryButtonText]}>Upload a Valid Image</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F5F5F5' },
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  title: { fontSize: 28, fontWeight: 'bold', color: '#333', marginBottom: 40 },
  button: {
    backgroundColor: '#007AFF', paddingVertical: 15, paddingHorizontal: 40,
    borderRadius: 10, elevation: 5, width: '80%', alignItems: 'center',
  },
  buttonText: { color: '#FFFFFF', fontSize: 18, fontWeight: '600' },
  separatorText: { marginVertical: 20, fontSize: 16, color: '#888' },
  secondaryButton: { backgroundColor: '#FFFFFF', borderWidth: 1, borderColor: '#007AFF' },
  secondaryButtonText: { color: '#007AFF' },
});